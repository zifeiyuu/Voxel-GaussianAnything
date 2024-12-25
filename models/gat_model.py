import torch
import torch.nn.functional as F
import logging
import time
import torch.nn as nn

from pathlib import Path
from einops import rearrange

from .encoder.layers import BackprojectDepth
from .encoder.dust3r_encoder import Dust3rEncoder
from .encoder.rgb_unidepth_encoder import Rgb_unidepth_Encoder
from .encoder.moge_encoder import MoGe_Encoder
from .encoder.voxel_encoder import prepare_hard_vfe_inputs_scatter_fast, HardVFE
from .decoder.gauss_util import focal2fov, getProjectionMatrix, K_to_NDC_pp, render_predicted
from .base_model import BaseModel
from .heads.gat_head import LinearHead
from misc.util import add_source_frame_id
from misc.depth import estimate_depth_scale, estimate_depth_scale_ransac, depthmap_to_absolute_camera_coordinates
from misc.visualise_3d import storePly # for debugging

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

class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

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
        
        self.vfe = HardVFE(in_channels=32+3+3, feat_channels=[64, 64], voxel_size=(0.02, 0.02, 0.02), point_cloud_range=(-5, -5, 0, 5, 5, 20))
        self.parameters_to_train += [{"params": self.vfe.parameters()}]


        self.use_decoder_3d = cfg.model.use_decoder_3d
        if self.use_decoder_3d:
            self.normalize_before_decoder_3d = cfg.model.normalize_before_decoder_3d
            self.decoder_3d = PointTransformerDecoder(cfg)
            self.parameters_to_train += self.decoder_3d.get_parameter_groups()
            head_in_dim = self.decoder_3d.transformer.out_dim

        self.decoder_gs = LinearHead(cfg, head_in_dim, xyz_scale=cfg.model.xyz_scale, xyz_bias=cfg.model.xyz_bias)

        # init a point decoder for compute point offset
        # if cfg.loss.soft_cd.weight > 0: # means we use cd loss here
        #     self.decoder_point = nn.Linear(self.decoder_3d.transformer.out_dim[-1] + 3, 3)
        #     nn.init.xavier_uniform_(self.decoder_point.weight, cfg.model.point_head_scale)
        #     nn.init.constant_(self.decoder_point.bias, cfg.model.point_head_bias)
            
        self.point_pre_voxle = 2
        self.point_offset = nn.ModuleList([MLP(self.decoder_3d.transformer.out_dim[-1], out_channels=3) for _ in range(self.point_pre_voxle)])
        self.predict_feature = nn.ModuleList([MLP(self.decoder_3d.transformer.out_dim[-1]) for _ in range(self.point_pre_voxle)])
        
        for mlp in self.point_offset:
            nn.init.constant_(mlp.fc2.weight, 0)
            nn.init.constant_(mlp.fc2.bias, 0)
            
        self.parameters_to_train += [{"params": self.point_offset.parameters()}]
        self.parameters_to_train += [{"params": self.predict_feature.parameters()}]
        

        self.parameters_to_train += self.decoder_gs.get_parameter_groups()

        self.use_checkpoint = False
        self.use_reentrant = False
        

    def forward(self, inputs):
        cfg = self.cfg
        # we predict points and associated features in 3d space directly
        # we do not use unprojection, so as camera intrinsics

        pts3d_origin, pts_feat, pts_rgb, pts_depth_origin, padding_select = self.encoder(inputs) # (B, N, 3), (B, N, C), (B, N, 3)

        if cfg.model.pre_downsample:
            pts3d_origin, pts_feat, pts_rgb = random_droping(pts3d_origin, pts_feat, pts_rgb, cfg.model.donsample_ratio)
        # Warning!!!!!! this code only support bs=1
        for b in range(pts3d_origin.shape[0]):
            features, num_points, coors, voxel_centers = prepare_hard_vfe_inputs_scatter_fast(pts3d_origin[b], pts_feat[b], pts_rgb[b], point_cloud_range=(-5, -5, 0, 5, 5, 20))

            # storePly("/home/maoyucheng/code/GaussianAnything2/debug_vox.ply", voxel_centers.detach().cpu(), torch.zeros_like(voxel_centers).detach().cpu())
            # storePly("/home/maoyucheng/code/GaussianAnything2/debug.ply", pts3d_origin[b].detach().cpu(), torch.zeros_like(pts3d_origin[b]).detach().cpu())
            # breakpoint()
            voxels_features = self.vfe(features, num_points, coors)
            
        # Warning!!!!!! Manually unsqueeze the btach size dim
        voxel_centers = voxel_centers.unsqueeze(0)
        voxels_features = voxels_features.unsqueeze(0)

        if self.use_decoder_3d:
            if self.normalize_before_decoder_3d:
                # normalize points in each batch
                mean = voxel_centers.mean(dim=1, keepdim=True) # (B, 1, 3)
                z_median = torch.median(voxel_centers[:, :, 2:], dim=1, keepdim=True)[0] # (B, 1)
                pts3d = (voxel_centers - mean) / (z_median + 1e-6) # (B, N, 3)
            else:
                pts3d = voxel_centers
            pts3d_ori, pts_feat_ori = self.decoder_3d(pts3d, voxels_features)
            
            # WARNING!!! No multiscale gaussian here!!
            # add predicted gaussian centroid offset with pts3d to get the final 3d centroids
            if self.use_decoder_3d and self.normalize_before_decoder_3d:
                # denormalize
                pts3d_ori = pts3d_ori[-1] * (z_median + 1e-6) + mean # (B, N, 3)
            
            recon_pts = pts3d_ori.clone()
            pts3d, pts_feat = [], []
            for i in range(self.point_pre_voxle):
                pts3d.append(pts3d_ori[-1] + self.point_offset[i](pts_feat_ori[-1]))
                pts_feat.append(self.predict_feature[i](pts_feat_ori[-1]))

            pts3d, pts_feat = torch.cat(pts3d, dim=1), torch.cat(pts_feat, dim=1)

            outputs = self.decoder_gs([pts_feat]) 

            
        #offset after normalize
        if cfg.model.predict_offset:
            for i in range(cfg.model.overall_padding):
                offset = outputs["gauss_offset"]
                offset = rearrange(offset, "b s c n -> b (s n) c", s=self.cfg.model.gaussians_per_pixel)
                pts3d = pts3d + offset

        pts3d_reshape = rearrange(pts3d, "b (s n) c -> b s c n", s=cfg.model.gaussians_per_pixel)
        outputs["gauss_means"] = pts3d_reshape

        if cfg.loss.soft_cd.weight > 0:
            outputs["recon_pts"] = recon_pts
            
        outputs[("depth", 0)] = pts_depth_origin

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

def random_droping(pts3d_origin, pts_feat, pts_rgb, ratio):
    B, N, _ = pts3d_origin.shape
    device = pts3d_origin.device
    mask = (torch.rand(N, device=device) > ratio)
    # breakpoint()
    return pts3d_origin[:, mask], pts_feat[:, mask], pts_rgb[:, mask]