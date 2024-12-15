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
from moge.utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift
import utils3d
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

        raw_img_h, raw_img_w = rgbs.shape[-2:]
        aspect_ratio = raw_img_w / raw_img_h

        patch_h, patch_w = raw_img_h // 14, raw_img_w // 14

        rgbs = (rgbs - self.moge.image_mean) / self.moge.image_std

        # Apply image transformation for DINOv2
        image_14 = F.interpolate(rgbs, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=False, antialias=True)

        # Get intermediate layers from the backbone
        mixed_precision = False
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=mixed_precision):
            features = self.moge.backbone.get_intermediate_layers(image_14, self.moge.intermediate_layers, return_class_token=True)

        # Predict points (and mask)
        output = self.moge.head(features, rgbs)
        if self.moge.split_head:
            points, mask = output
        else:
            points, mask = output.split([3, 1], dim=1)
        points, mask = points.permute(0, 2, 3, 1), mask.squeeze(1)

        if self.moge.remap_output == 'linear' or self.moge.remap_output == False:
            pass
        elif self.moge.remap_output =='sinh' or self.moge.remap_output == True:
            points = torch.sinh(points)
        elif self.moge.remap_output == 'exp':
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.moge.remap_output =='sinh_exp':
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.moge.remap_output}")

        # Get camera-space point map. (Focal here is the focal length relative to half the image diagonal)
        focal, shift = recover_focal_shift(points, None if mask is None else mask > 0.5)
        fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
        fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 
        depth = points[..., 2] + shift[..., None, None]

        #apply mask
        # mask_binary = (depth > 0) & (mask > 0.5)
        # depth = torch.where(mask_binary, depth, torch.inf)

        mask = mask > 0.5  #(B C) H W

        pts_depth = rearrange(depth, '(b c) h w -> b (h w) c', b=B) #B (H W) C
        mask = rearrange(mask, '(b c) h w -> b (h w) c', b=B) #B (H W) C

        #fill the inf depth with a desgined value
        # max_depth = pts_depth[torch.isfinite(pts_depth)].max().item()  # Get the maximum valid depth
        # pts_depth = pts_depth.masked_fill(~mask.bool(), 1 + max_depth)

        # scale depth map smaller to avoid bug
        pts_depth = pts_depth * 0.5

        # vit encoder to get per-image features
        # Encode
        encoder_outputs = features[-1][0] #last layer of transformer    
        cls_token =  features[-1][1]
        encoder_outputs = (encoder_outputs + cls_token.unsqueeze(1)).contiguous()
        
        pts_feat = self.pts_feat_head(encoder_outputs)  # (B, H / P * W / P, EMBED_DIM) ->(B, H / P * W / P, p^2*D)
        pts_feat = rearrange(pts_feat, 'b hpwp (p d) -> b (hpwp p) d', p=self.patch_size**2, d=self.pts_feat_dim)               
        
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
