import torch
import logging
import time
import torch.nn as nn

from pathlib import Path
from einops import rearrange

from .encoder.layers import BackprojectDepth
from .encoder.dust3r_encoder import Dust3rEncoder
from .encoder.rgb_unidepth_encoder import Rgb_unidepth_Encoder
from .encoder.moge_encoder import MoGe_Encoder
from .decoder.gauss_util import focal2fov, getProjectionMatrix, K_to_NDC_pp, render_predicted
from .base_model import BaseModel
from .heads.gat_head import LinearHead
from misc.util import add_source_frame_id
from misc.depth import estimate_depth_scale, estimate_depth_scale_ransac

from .decoder.pointcloud_decoder import PointTransformerDecoder

from IPython import embed
from models.decoder.resnet_decoder import ResnetDecoder

from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_mean

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
        elif "moge" in cfg.model.backbone.name:
            self.encoder = MoGe_Encoder(cfg)
            
        self.parameters_to_train += self.encoder.get_parameter_groups()
        head_in_dim = [cfg.model.backbone.pts_feat_dim]

        self.use_decoder_3d = cfg.model.use_decoder_3d
        if self.use_decoder_3d:
            self.normalize_before_decoder_3d = cfg.model.normalize_before_decoder_3d
            self.decoder_3d = PointTransformerDecoder(cfg)
            self.parameters_to_train += self.decoder_3d.get_parameter_groups()
            head_in_dim = self.decoder_3d.transformer.out_dim

        self.decoder_gs = LinearHead(cfg, head_in_dim)
        self.parameters_to_train += self.decoder_gs.get_parameter_groups()

        self.use_checkpoint = False
        self.use_reentrant = False

    def forward(self, inputs):
        cfg = self.cfg

        # we predict points and associated features in 3d space directly
        # we do not use unprojection, so as camera intrinsics

        pts3d_origin, pts_feat, pts_rgb, pts_depth = self.encoder(inputs) # (B, N, 3), (B, N, C), (B, N, 3)

        if cfg.model.pre_downsample:
            pts3d_origin, pts_feat, pts_rgb = points_to_voxels(pts3d_origin, pts_feat, pts_rgb, cfg.model.voxel_size)

        if self.use_decoder_3d:
            if self.normalize_before_decoder_3d:
                # normalize points in each batch
                mean = pts3d_origin.mean(dim=1, keepdim=True) # (B, 1, 3)
                z_median = torch.median(pts3d_origin[:, :, 2:], dim=1, keepdim=True)[0] # (B, 1)
                pts3d = (pts3d_origin - mean) / (z_median + 1e-6) # (B, N, 3)
            else:
                pts3d = pts3d_origin

            pts3d, pts_feat = self.decoder_3d(pts3d, torch.cat([pts_rgb, pts_feat], dim=-1))
            pts3d = torch.cat(pts3d, dim=1)

            # predict gaussian parameters for each point
            outputs = self.decoder_gs(pts_feat)
        else:
            mean = pts3d_origin.mean(dim=1, keepdim=True) # (B, 1, 3)
            z_median = torch.median(pts3d_origin[:, :, 2:], dim=1, keepdim=True)[0] # (B, 1)
            pts3d_normed = (pts3d_origin - mean) / (z_median + 1e-6) # (B, N, 3)
            pts3d = pts3d_origin

            # predict gaussian parameters for each point
            outputs = self.decoder_gs([pts_feat])

        # add predicted gaussian centroid offset with pts3d to get the final 3d centroids
        if self.use_decoder_3d and self.normalize_before_decoder_3d:
            # denormalize
            pts3d = pts3d * (z_median + 1e-6) + mean # (B, N, 3)

        #offset after normalize
        if cfg.model.predict_offset:
            offset = outputs["gauss_offset"]
            offset = rearrange(offset, "b s c n -> b (s n) c", s=cfg.model.gaussians_per_pixel)
            pts3d = pts3d + offset

        # pts3d_origin = rearrange(pts3d_origin, "b (s n) c -> b s c n", s=cfg.model.gaussians_per_pixel)
        # outputs["gauss_means_origin"] = pts3d_origin
        pts3d_reshape = rearrange(pts3d, "b (s n) c -> b s c n", s=cfg.model.gaussians_per_pixel)
        outputs["gauss_means"] = pts3d_reshape
        outputs[("depth", 0)] = pts_depth

        if cfg.model.gaussian_rendering:
            self.process_gt_poses(inputs, outputs)
            self.render_images(inputs, outputs)

        return outputs


def points_to_voxels(pts3d_origin, pts_feat, pts_rgb, voxel_size):
    B, N, _ = pts3d_origin.shape
    feature_dim = pts_feat.shape[2]
    voxel_xyz, voxel_features, voxel_colors = [], [], []

    for b in range(B):
        # Quantize points into voxel indices
        voxel_indices = (pts3d_origin[b] / voxel_size).floor().long()
        aggregated_xyz, inverse_indices = torch.unique(voxel_indices, dim=0, return_inverse=True)
        
        # Aggregate features and colors separately
        aggregated_features = scatter_mean(pts_feat[b], inverse_indices, dim=0)
        aggregated_colors = scatter_mean(pts_rgb[b], inverse_indices, dim=0)

        voxel_xyz.append(aggregated_xyz)
        voxel_features.append(aggregated_features)
        voxel_colors.append(aggregated_colors)

    # Stack results
    max_voxels = max(loc.shape[0] for loc in voxel_xyz)
    padded_xyz = torch.zeros(B, max_voxels, 3, dtype=torch.long)
    padded_features = torch.zeros(B, max_voxels, feature_dim)
    padded_colors = torch.zeros(B, max_voxels, 3)

    for b in range(B):
        M = voxel_xyz[b].shape[0]
        padded_xyz[b, :M] = voxel_xyz[b]
        padded_features[b, :M] = voxel_features[b]
        padded_colors[b, :M] = voxel_colors[b]

    return padded_xyz.float().to(pts3d_origin.device), padded_features.to(pts_feat.device), padded_colors.to(pts_rgb.device)