import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
import torchvision.transforms.functional as TF
from math import ceil

from einops import rearrange
from .resnet_encoder import ResnetEncoder
from ..decoder.resnet_decoder import ResnetDecoder, ResnetDepthDecoder
from IPython import embed
from .geometric import generate_rays
from .helper import freeze_all_params, _paddings, _shapes, _preprocess, _postprocess
from .layers import BackprojectDepth

import torch.distributed as dist
import math
import cv2
import time

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
moge_path = os.path.join(base_dir, 'submodules/MoGe')
sys.path.insert(0, moge_path)
from moge.model import MoGeModel
sys.path.pop(0)

class MoGe_Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.patch_size = 14  #hard code!! to fit dino v2 l14, don't know how to change patch size inside. default patch size in dust3r encoder is 16 != 14
        self.pts_feat_dim = cfg.model.backbone.pts_feat_dim

        if dist.is_initialized():
            local_rank = dist.get_rank()
            self.device=f'cuda:{local_rank}'
            self.moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(self.device)   
        else:
            self.device='cuda:0'
            self.moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(self.device)
        # self.moge.image_shape = [cfg.dataset.height, cfg.dataset.width]
        print("MoGe loaded!")
        self.set_backproject()

        self.parameters_to_train = []
        all_moge_modules = [
            "backbone",
            "head"
        ]
        modules_to_delete_moge = []
        modules_to_freeze_moge = []

        self.enc_dim = self.moge.backbone.embed_dim  
      
        self.pts_feat_head = nn.Sequential(
            nn.Linear(self.enc_dim, self.pts_feat_dim * self.patch_size**2)
        )        
        self.parameters_to_train += [{"params": list(self.pts_feat_head.parameters())}]

        # freeze ,delete and add modules
        if cfg.model.backbone.moge.freeze_encoder:
            modules_to_freeze_moge += ["backbone"]
        if cfg.model.backbone.moge.freeze_decoder:
            modules_to_freeze_moge += ["head"]

        for module in all_moge_modules:
            if module in modules_to_delete_moge:
                delattr(self.moge, module)
            elif module in modules_to_freeze_moge:
                freeze_all_params(getattr(self.moge, module))
            else:
                self.parameters_to_train += [{"params": list(getattr(self.moge, module).parameters())}]

    def get_parameter_groups(self):
        return self.parameters_to_train
    
    def set_backproject(self):
        cfg = self.cfg
        backproject_depth = {}
        H_dataset = cfg.dataset.height
        W_dataset = cfg.dataset.width
        for scale in cfg.model.scales:
            h = H_dataset // (2 ** scale)
            w = W_dataset // (2 ** scale)
            if cfg.model.shift_rays_half_pixel == "zero":
                shift_rays_half_pixel = 0
            elif cfg.model.shift_rays_half_pixel == "forward":
                shift_rays_half_pixel = 0.5
            elif cfg.model.shift_rays_half_pixel == "backward":
                shift_rays_half_pixel = -0.5
            else:
                raise NotImplementedError
            backproject_depth[str(scale)] = BackprojectDepth(
                cfg.data_loader.batch_size * cfg.model.gaussians_per_pixel, 
                # backprojection can be different if padding was used
                h + 2 * self.cfg.dataset.pad_border_aug, 
                w + 2 * self.cfg.dataset.pad_border_aug,
                shift_rays_half_pixel=shift_rays_half_pixel
            )
        self.backproject_depth = nn.ModuleDict(backproject_depth)


    def forward(self, inputs):
        #rgb
        rgbs = inputs["color_aug", 0, 0]  
        #use gt intrinsics
        gt_K = inputs[("K_src", 0)]
        inv_K = torch.linalg.inv(gt_K.float())
        if rgbs.ndim == 3:
            rgbs = rgbs.unsqueeze(0)
        if gt_K is not None and gt_K.ndim == 2:
            gt_K = gt_K.unsqueeze(0)
        B, C, H, W = rgbs.shape      

        # Infer 
        predictions = self.moge.infer(rgbs)

        pts_depth = predictions['depth']  #(B C) H W
        pts_depth = rearrange(pts_depth, '(b c) h w -> b (h w) c', b=B) #B (H W) C
        mask = predictions['mask']  #(B C) H W
        mask = rearrange(mask, '(b c) h w -> b (h w) c', b=B) #B (H W) C

        #fill the inf depth with a desgined value
        max_depth = pts_depth[torch.isfinite(pts_depth)].max().item()  # Get the maximum valid depth
        pts_depth = pts_depth.masked_fill(~mask.bool(), 1 + max_depth)

        # vit encoder to get per-image features
        # Encode
        encoder_outputs = self.moge.features[-1][0] #last layer of transformer     (B X D)   I DON'T KNOW WHAT DOES X MEANS????
        feat_num = encoder_outputs.shape[1]

        # Find closest dimensions  TRICKY APPROACH.
        for i in range(int(math.sqrt(feat_num)), 0, -1):
            if feat_num % i == 0:
                hx, wx = i, feat_num // i
                break
        # Reshape to add spatial dimensions
        encoder_outputs = rearrange(encoder_outputs, 'b (hx wx) e-> b hx wx e', hx=hx, wx=wx)  
        # Downsample using F.interpolate
        encoder_outputs = encoder_outputs.permute(0, 3, 1, 2)
        encoder_outputs = F.interpolate(encoder_outputs, size=(int(H / self.patch_size), int(W / self.patch_size)), mode='bilinear') 
        encoder_outputs = encoder_outputs.permute(0, 2, 3, 1) # B, H/P, W/P, EMBED_DIM

        pts_feat = self.pts_feat_head(encoder_outputs)  # (B, H / P, W / P, EMBED_DIM) ->(B, H / P, W / P, p^2*D)
        pts_feat = rearrange(pts_feat, 'b hp wp (p d) -> b (hp wp p) d', p=self.patch_size**2, d=self.pts_feat_dim)               
        
        # back project depth to world splace
        scale = self.cfg.model.scales[0]
        pts3d = self.backproject_depth[str(scale)](pts_depth, inv_K)
        pts3d = rearrange(pts3d, 'b c n -> b n c')[:, :, :3]

        #directly give decoder rgb information for each 3d point
        pts_rgb = rearrange(rgbs, 'b c h w -> b (h w) c')

        # # mask out invalid points
        # if torch.count_nonzero(mask).item() != H*W:
        #     D = self.pts_feat_dim
        #     # Apply mask directly
        #     with torch.no_grad():
        #         pts_feat = pts_feat[mask.expand(-1, -1, D)].view(B, -1, D)
        #         pts3d = pts3d[mask.expand(-1, -1, 3)].view(B, -1, 3)
        #         pts_rgb = pts_rgb[mask.expand(-1, -1, 3)].view(B, -1, 3)

        return pts3d, pts_feat, pts_rgb
