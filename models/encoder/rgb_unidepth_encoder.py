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
from .layers import BackprojectDepth
from .geometric import generate_rays

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
unidepth_path = os.path.join(base_dir, 'submodules/UniDepth')
sys.path.insert(0, unidepth_path)
from hubconf import UniDepth
from unidepth.utils.constants import (IMAGENET_DATASET_MEAN,
                                      IMAGENET_DATASET_STD)
sys.path.pop(0)

def freeze_all_params(modules):
    if isinstance(modules, list) or isinstance(modules, tuple):
        for module in modules:
            try:
                for n, param in module.named_parameters():
                    param.requires_grad = False
            except AttributeError:
                # module is directly a parameter
                module.requires_grad = False
    else:
        assert isinstance(modules, nn.Module)
        try:
            for n, param in modules.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            modules.requires_grad = False

# inference helpers
def _paddings(image_shape, network_shape):
    cur_h, cur_w = image_shape
    h, w = network_shape
    pad_top, pad_bottom = (h - cur_h) // 2, h - cur_h - (h - cur_h) // 2
    pad_left, pad_right = (w - cur_w) // 2, w - cur_w - (w - cur_w) // 2
    return pad_left, pad_right, pad_top, pad_bottom


def _shapes(image_shape, network_shape):
    h, w = image_shape
    input_ratio = w / h
    output_ratio = network_shape[1] / network_shape[0]
    if output_ratio > input_ratio:
        ratio = network_shape[0] / h
    elif output_ratio <= input_ratio:
        ratio = network_shape[1] / w
    return (ceil(h * ratio - 0.5), ceil(w * ratio - 0.5)), ratio


def _preprocess(rgbs, intrinsics, shapes, pads, ratio, output_shapes):
    (pad_left, pad_right, pad_top, pad_bottom) = pads
    rgbs = F.interpolate(
        rgbs, size=shapes, mode="bilinear", align_corners=False, antialias=True
    )
    rgbs = F.pad(rgbs, (pad_left, pad_right, pad_top, pad_bottom), mode="constant")
    if intrinsics is not None:
        intrinsics = intrinsics.clone()
        intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * ratio
        intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * ratio
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * ratio + pad_left
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * ratio + pad_top
        return rgbs, intrinsics
    return rgbs, None

def _postprocess(predictions, intrinsics, shapes, pads, ratio, original_shapes):
    (pad_left, pad_right, pad_top, pad_bottom) = pads
    # pred mean, trim paddings, and upsample to input dim
    predictions = sum(
        [
            F.interpolate(
                x.clone(),
                size=shapes,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            for x in predictions
        ]
    ) / len(predictions)

    predictions = predictions[
        ..., pad_top : shapes[0] - pad_bottom, pad_left : shapes[1] - pad_right
    ]
    predictions = F.interpolate(
        predictions,
        size=original_shapes,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    intrinsics[:, 0, 0] = intrinsics[:, 0, 0] / ratio
    intrinsics[:, 1, 1] = intrinsics[:, 1, 1] / ratio
    intrinsics[:, 0, 2] = (intrinsics[:, 0, 2] - pad_left) / ratio
    intrinsics[:, 1, 2] = (intrinsics[:, 1, 2] - pad_top) / ratio
    return predictions, intrinsics



class Rgb_unidepth_Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.use_unidepth_decoder = cfg.model.backbone.use_unidepth_decoder

        self.unidepth = UniDepth(
            version=cfg.model.depth.version, 
            backbone=cfg.model.depth.backbone, 
            pretrained=True
        )
        # self.unidepth.image_shape = [cfg.dataset.height, cfg.dataset.width]
        
        print("Unidepth_v1 loaded!")
        self.set_backproject()

        self.parameters_to_train = []
        all_unidepth_modules = [
            "pixel_encoder",
            "pixel_decoder"
        ]
        modules_to_delete = []
        modules_to_freeze = []

        self.pts_feat_dim = cfg.model.backbone.pts_feat_dim
        self.patch_size = self.unidepth.pixel_encoder.patch_size
        enc_dim = self.unidepth.pixel_encoder.embed_dim

        if not self.use_unidepth_decoder:
            modules_to_delete += ["pixel_decoder"]

            self.pts_head = nn.Sequential(
                nn.Linear(enc_dim, 1 * self.patch_size**2)
            )
            self.parameters_to_train += [{"params": list(self.pts_head.parameters())}]

        self.pts_feat_head = nn.Sequential(
            nn.Linear(enc_dim, self.pts_feat_dim * self.patch_size**2)
        )
        self.parameters_to_train += [{"params": list(self.pts_feat_head.parameters())}]

        # freeze ,delete and add modules
        if cfg.model.backbone.freeze_encoder:
            modules_to_freeze += ["pixel_encoder"]
        if cfg.model.backbone.freeze_decoder:
            modules_to_freeze += ["pixel_decoder"]
        if cfg.model.backbone.freeze_head:
            modules_to_freeze += []

        for module in all_unidepth_modules:
            if module in modules_to_delete:
                delattr(self.unidepth, module)
            elif module in modules_to_freeze:
                freeze_all_params(getattr(self.unidepth, module))
            else:
                self.parameters_to_train += [{"params": list(getattr(self.unidepth, module).parameters())}]

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

    def preprocess(self, rgbs, gt_K, pad_left, pad_right, pad_top, pad_bottom):
        B, _, H, W = rgbs.shape

        # process image and intrinsiscs (if any) to match network input (slow?)
        if rgbs.max() > 5 or rgbs.dtype == torch.uint8:
            rgbs = rgbs.to(torch.float32).div(255)
        if rgbs.min() >= 0.0 and rgbs.max() <= 1.0:
            rgbs = TF.normalize(
                rgbs,
                mean=IMAGENET_DATASET_MEAN,
                std=IMAGENET_DATASET_STD,
            )

        (h, w), ratio = _shapes((H, W), self.unidepth.image_shape)
        rgbs, gt_K = _preprocess(
            rgbs,
            gt_K,
            (h, w),
            (pad_left, pad_right, pad_top, pad_bottom),
            ratio,
            self.unidepth.image_shape,
        )
        return rgbs, gt_K

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
        B, _, H, W = rgbs.shape
        
        #if want to reshape
        (h, w), ratio = _shapes((H, W), self.unidepth.image_shape)
        pad_left, pad_right, pad_top, pad_bottom = _paddings((h, w), self.unidepth.image_shape)

        # rgbs, gt_K = self.preprocess(rgbs, gt_K, pad_left, pad_right, pad_top, pad_bottom)

        # vit encoder to get per-image features
        # Encode
        original_encoder_outputs, cls_tokens = self.unidepth.pixel_encoder(rgbs) #list[torch.Size([3, 462 / 14, 616 / 14, 1024])], len = 24, (B, H / P, W / P, EMBED_DIM)
        if "dino" in self.unidepth.pixel_encoder.__class__.__name__.lower():
            original_encoder_outputs = [
                (x + y.unsqueeze(1)).contiguous()
                for x, y in zip(original_encoder_outputs, cls_tokens)
            ]

        #from list to single tensor##?
        encoder_outputs = [x.contiguous() for x in original_encoder_outputs]
        if len(encoder_outputs) == 1:
            encoder_outputs = encoder_outputs[0]
        else:
            encoder_outputs = torch.stack(encoder_outputs, dim=-1).max(dim=-1).values  #(B, H / P, W / P, EMBED_DIM)

        if self.use_unidepth_decoder:
            inputs = {}
            inputs["image"] = rgbs
            inputs["encoder_outputs"] = original_encoder_outputs
            inputs["cls_tokens"] = cls_tokens
            # Get camera infos
            rays, angles = generate_rays(
                gt_K, self.unidepth.image_shape, noisy=self.training
            )
            inputs["rays"] = rays
            inputs["angles"] = angles
            inputs["K"] = gt_K
            self.unidepth.pixel_decoder.test_fixed_camera = True  # use GT camera in fwd

            # decode all, get the feature of the final layer of the decoder
            pred_intrinsics, predictions, _ = self.unidepth.pixel_decoder(inputs, {})  

            # predictions, pred_intrinsics = _postprocess(
            #     predictions,
            #     pred_intrinsics,
            #     self.unidepth.image_shape,
            #     (pad_left, pad_right, pad_top, pad_bottom),
            #     ratio,
            #     (H, W),
            # )

            predictions = sum(
                [
                    F.interpolate(
                        x.clone(),
                        size=self.unidepth.image_shape,
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )
                    for x in predictions
                ]
            ) / len(predictions)

            # Output data, use for loss computation
            outputs = {
                "depth": predictions[:, -1:],                
                "intrinsics": pred_intrinsics
            }
            self.unidepth.pixel_decoder.test_fixed_camera = False

            pts_depth, pred_K = outputs["depth"], outputs["intrinsics"]  #B C H W
            pts_depth = rearrange(pts_depth, 'b c h w -> b (h w) c') #B (H W) C

            pts_feat = self.pts_feat_head(encoder_outputs)  # (B, H / P, W / P, EMBED_DIM) ->(B, H / P, W / P, p^2*D)
            pts_feat = rearrange(pts_feat, 'b hp wp (p d) -> b (hp wp p) d', p=self.patch_size**2, d=self.pts_feat_dim)

        else:
            # linear layer to decode
            pts_depth = self.pts_head(encoder_outputs) # (B, H / P, W / P, EMBED_DIM) ->(B, H / P, W / P, p^2*1)
            pts_depth = rearrange(pts_depth, 'b hp wp (p d) -> b (hp wp p) d', p=self.patch_size**2, d=1)
            pts_feat = self.pts_feat_head(encoder_outputs) # (B, H / P, W / P, EMBED_DIM) ->(B, H / P, W / P, p^2*D)
            pts_feat = rearrange(pts_feat, 'b hp wp (p d) -> b (hp wp p) d', p=self.patch_size**2, d=self.pts_feat_dim)


        # back project depth to world splace
        scale = self.cfg.model.scales[0]
        pts3d = self.backproject_depth[str(scale)](pts_depth, inv_K)
        pts3d = rearrange(pts3d, 'b c n -> b n c')[:, :, :3]

        #directly give decoder rgb information for each 3d point
        pts_rgb = rearrange(rgbs, 'b c h w -> b (h w) c')

        return pts3d, pts_feat, pts_rgb #,rgbs #need original color information??
