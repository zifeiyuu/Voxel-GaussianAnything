import torch
import logging
import time
import torch.nn as nn

from pathlib import Path
from einops import rearrange

from .encoder.layers import BackprojectDepth
from .encoder.dust3r_encoder import Dust3rEncoder
from .encoder.rgb_unidepth_encoder import Rgb_unidepth_Encoder
from .decoder.gauss_util import focal2fov, getProjectionMatrix, K_to_NDC_pp, render_predicted
from .base_model import BaseModel
from .heads.gat_head import LinearHead
from misc.util import add_source_frame_id
from misc.depth import estimate_depth_scale, estimate_depth_scale_ransac

from .decoder.pointcloud_decoder import PointTransformerDecoder

from IPython import embed
from models.decoder.resnet_decoder import ResnetDecoder

from torch.utils.checkpoint import checkpoint

def default_param_group(model):
    return [{'params': model.parameters()}]


def to_device(inputs, device):
    for key, ipt in inputs.items():
        if isinstance(ipt, torch.Tensor):
            inputs[key] = ipt.to(device)
    return inputs

class GATModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.cfg = cfg

        self.parameters_to_train = []

        # define the model
        if "dust3r" in cfg.model.backbone.name:
            self.encoder = Dust3rEncoder(cfg)
        elif "unidepth" in cfg.model.backbone.name:
            self.encoder = Rgb_unidepth_Encoder(cfg)
            
        self.parameters_to_train += self.encoder.get_parameter_groups()

        self.use_decoder_3d = cfg.model.use_decoder_3d
        if self.use_decoder_3d:
            self.normalize_before_decoder_3d = cfg.model.normalize_before_decoder_3d
            self.decoder_3d = PointTransformerDecoder(cfg)
            self.parameters_to_train += self.decoder_3d.get_parameter_groups()

        self.decoder_gs = LinearHead(cfg)
        self.parameters_to_train += self.decoder_gs.get_parameter_groups()

        self.use_checkpoint = False
        self.use_reentrant = False

    def forward(self, inputs):
        cfg = self.cfg

        # we predict points and associated features in 3d space directly
        # we do not use unprojection, so as camera intrinsics

        if self.use_checkpoint:
            def run_encoder(inputs):
                return self.encoder(inputs)
            pts3d_origin, pts_feat, pts_rgb = checkpoint(run_encoder, inputs, use_reentrant=self.use_reentrant) # (B, N, 3), (B, N, C), (B, N, 3)
        else:
            pts3d_origin, pts_feat, pts_rgb = self.encoder(inputs) # (B, N, 3), (B, N, C), (B, N, 3)

        if self.use_decoder_3d:
            if self.normalize_before_decoder_3d:
                # normalize points in each batch
                mean = pts3d_origin.mean(dim=1, keepdim=True) # (B, 1, 3)
                z_median = torch.median(pts3d_origin[:, :, 2:], dim=1, keepdim=True)[0] # (B, 1)
                pts3d = (pts3d_origin - mean) / (z_median + 1e-6) # (B, N, 3)
            else:
                pts3d = pts3d_origin

            
            if self.use_checkpoint:
                def run_decoder_3d(pts3d, pts_rgb, pts_feat):
                    return self.decoder_3d(pts3d, torch.cat([pts_rgb, pts_feat], dim=-1))
                pts3d, pts_feat = checkpoint(
                    run_decoder_3d,
                    pts3d,
                    pts_rgb, 
                    pts_feat,
                    use_reentrant=self.use_reentrant
                )
            else:
                pts3d, pts_feat = self.decoder_3d(pts3d, torch.cat([pts_rgb, pts_feat], dim=-1))

            # predict gaussian parameters for each point
            if self.use_checkpoint:
                def run_decoder_gs(pts_feat):
                    return self.decoder_gs(torch.cat([pts_feat], dim=-1))
                outputs = checkpoint(
                    run_decoder_gs,
                    pts_feat,
                    use_reentrant=self.use_reentrant
                )
            else:
                outputs = self.decoder_gs(torch.cat([pts_feat], dim=-1))
        else:
            mean = pts3d_origin.mean(dim=1, keepdim=True) # (B, 1, 3)
            z_median = torch.median(pts3d_origin[:, :, 2:], dim=1, keepdim=True)[0] # (B, 1)
            pts3d_normed = (pts3d_origin - mean) / (z_median + 1e-6) # (B, N, 3)
            pts3d = pts3d_origin

            # predict gaussian parameters for each point
            if self.use_checkpoint:
                def run_decoder_gs(pts_feat):
                    return self.decoder_gs(torch.cat([pts_feat], dim=-1))
                outputs = checkpoint(
                    run_decoder_gs,
                    pts_feat,
                    use_reentrant=self.use_reentrant
                )
            else:
                outputs = self.decoder_gs(torch.cat([pts_feat], dim=-1))

        # add predicted gaussian centroid offset with pts3d to get the final 3d centroids
        if self.use_decoder_3d and self.normalize_before_decoder_3d:
            # denormalize
            pts3d = pts3d * (z_median + 1e-6) + mean # (B, N, 3)

        pts3d_reshape = rearrange(pts3d, "b (s n) c -> b s c n", s=cfg.model.gaussians_per_pixel)
        pts3d_origin = rearrange(pts3d_origin, "b (s n) c -> b s c n", s=cfg.model.gaussians_per_pixel)
        outputs["gauss_means_origin"] = pts3d_origin
        outputs["gauss_means"] = pts3d_reshape[:, :, :3, :]

        if cfg.model.gaussian_rendering:
            self.process_gt_poses(inputs, outputs)
            self.render_images(inputs, outputs)

        return outputs

