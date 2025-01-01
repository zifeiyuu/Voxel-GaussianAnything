import torch
import torch.nn.functional as F
import logging
import time
import torch.nn as nn

from pathlib import Path
from einops import rearrange
import collections

from .encoder.moge_encoder import MoGe_MVEncoder
from .encoder.voxel_encoder import prepare_hard_vfe_inputs_scatter_fast, HardVFE, compute_voxel_coors_and_centers
from .decoder.gauss_util import focal2fov, getProjectionMatrix, K_to_NDC_pp, render_predicted
from .base_model import BaseModel
from .heads.gat_head import LinearHead
from misc.util import add_source_frame_id
from misc.depth import estimate_depth_scale, estimate_depth_scale_ransac, depthmap_to_absolute_camera_coordinates_torch
from misc.visualise_3d import storePly # for debugging
import numpy as np

from .decoder.pointcloud_decoder import PointTransformerDecoder
from .decoder.voxel_predictor import VoxPredictor

from IPython import embed

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
        self.using_frames = [0] + cfg.model.gauss_novel_frames
        self.pretrain_parameters = []
        self.parameters_to_train = []

        # define the model
        if "moge" in cfg.model.backbone.name:
            self.encoder = MoGe_MVEncoder(cfg)
            
        self.parameters_to_train += self.encoder.get_parameter_groups()
        head_in_dim = [cfg.model.backbone.pts_feat_dim]

        self.voxel_size, self.pc_range, self.coarse_voxel_size = cfg.model.voxel_size, cfg.model.pc_range, cfg.model.coarse_voxel_size
        self.vfe = HardVFE(in_channels=cfg.model.backbone.pts_feat_dim+3+3, feat_channels=[64, 64], voxel_size=(self.voxel_size, self.voxel_size, self.voxel_size), point_cloud_range=self.pc_range)
        self.parameters_to_train += [{"params": self.vfe.parameters()}]
        self.pretrain_parameters += [{"params": self.vfe.parameters()}]
        
        self.vox_pred = VoxPredictor(cfg)
        self.parameters_to_train += self.vox_pred.get_parameter_groups()
        self.pretrain_parameters += self.vox_pred.get_parameter_groups()

        self.decoder_3d = PointTransformerDecoder(cfg)
        self.parameters_to_train += self.decoder_3d.get_parameter_groups()
        head_in_dim = self.decoder_3d.transformer.out_dim

        self.decoder_gs = LinearHead(cfg, head_in_dim, xyz_scale=cfg.model.xyz_scale, xyz_bias=cfg.model.xyz_bias)
        self.parameters_to_train += self.decoder_gs.get_parameter_groups()
        
        self.mask_token = nn.Parameter(torch.zeros(1, 64), requires_grad=True)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.parameters_to_train += [{"params": self.mask_token}]


        self.use_checkpoint = False
        self.use_reentrant = False
        
    def padding_voxel_maxpooling(self, coors, padded_voxels):
        # WARNING! This function only take bs=1 inout, i.e., the first dim of coords should always be zero
        assert coors[..., 0].sum() == 0
        canvs_size = (int(abs(self.pc_range[0] - self.pc_range[3]) / self.voxel_size), int(abs(self.pc_range[1] - self.pc_range[4]) / self.voxel_size), int(abs(self.pc_range[2] - self.pc_range[5]) / self.voxel_size))
        binary_canvas = torch.zeros(1, 1, canvs_size[2], canvs_size[1], canvs_size[0], dtype=coors.dtype, device=coors.device)
        binary_canvas[:, :, coors[:,1], coors[:,2], coors[:,3]] = 1
        
        pooled_binary_canvs = F.interpolate(padded_voxels.unsqueeze(1), size=binary_canvas.shape[2:], mode="nearest").int().squeeze(1)
        padded_binary_canvs = (pooled_binary_canvs - binary_canvas.squeeze(1)).clamp(min=0)
        padded_coors = torch.nonzero(padded_binary_canvs == 1, as_tuple=False)

        grid_min = self.pc_range[:3]
        voxel_centers = torch.stack([
            padded_coors[..., -1].float() * self.voxel_size + grid_min[0] + self.voxel_size / 2,
            padded_coors[..., -2].float() * self.voxel_size + grid_min[1] + self.voxel_size / 2,
            padded_coors[..., -3].float() * self.voxel_size + grid_min[2] + self.voxel_size / 2
        ], dim=1)  # [M, 3]
        
        return padded_coors, voxel_centers
    
    def get_projected_points(self, inputs, outputs):
        # we project points from target view and novel view
        pts3d_batch = []

        for batch_idx in range(len(inputs[('depth_sparse', 0)])):    # outputs[('depth_pred', 0)]    inputs[('depth_sparse', 0)]
            pts3d = []
            for frameid in self.using_frames:
                depth, K = inputs[('depth_sparse', frameid)][batch_idx], inputs[('K_tgt', frameid)][batch_idx]
                c2w = outputs[('cam_T_cam', frameid, 0)][batch_idx] if frameid != 0 else None

                _pts3d, mask = depthmap_to_absolute_camera_coordinates_torch(depth, K, c2w)
                pts3d.append(_pts3d[mask])
            pts3d = torch.cat(pts3d)
            pts3d_batch.append(pts3d)
        return pts3d_batch
    
    def get_projected_points_nomask(self, inputs, outputs):
        # we project points from target view and novel view
        pts3d_batch = []

        for batch_idx in range(len(inputs[('depth_sparse', 0)])):    # outputs[('depth_pred', 0)]    inputs[('depth_sparse', 0)]
            pts3d = []
            for frameid in self.using_frames:
                depth, K = inputs[('depth_sparse', frameid)][batch_idx], inputs[('K_tgt', frameid)][batch_idx]
                c2w = outputs[('cam_T_cam', frameid, 0)][batch_idx] if frameid != 0 else None

                _pts3d, mask = depthmap_to_absolute_camera_coordinates_torch(depth, K, c2w)
                pts3d.append(rearrange(_pts3d, "h w c -> (h w) c"))
            pts3d = torch.cat(pts3d)
            pts3d_batch.append(pts3d)
        return pts3d_batch
    
    def get_projected_points_source_view(self, inputs, outputs):
        # project points from source view
        pts3d_batch = []

        for batch_idx in range(len(inputs[('depth_sparse', 0)])):    # outputs[('depth_pred', 0)]    inputs[('depth_sparse', 0)]
            depth, K = inputs[('depth_sparse', 0)][batch_idx], inputs[('K_tgt', 0)][batch_idx]
            c2w = None
            _pts3d, mask = depthmap_to_absolute_camera_coordinates_torch(depth, K, c2w)
            pts3d_batch.append(_pts3d[mask])
        return pts3d_batch
    
    def get_projected_points_source_view_nomask(self, inputs, outputs):
        # project points from source view
        pts3d_batch = []

        for batch_idx in range(len(inputs[('depth_sparse', 0)])):    # outputs[('depth_pred', 0)]    inputs[('depth_sparse', 0)]
            depth, K = inputs[('depth_sparse', 0)][batch_idx], inputs[('K_tgt', 0)][batch_idx]
            c2w = None
            _pts3d, mask = depthmap_to_absolute_camera_coordinates_torch(depth, K, c2w)
            pts3d_batch.append(rearrange(_pts3d, "h w c -> (h w) c"))
        return pts3d_batch
    
    def get_binary_voxels(self, points):
        coors, voxel_centers = compute_voxel_coors_and_centers(points, voxel_size=self.coarse_voxel_size, point_cloud_range=self.pc_range)
        
        canvs_size = (int(abs(self.pc_range[0] - self.pc_range[3]) / self.coarse_voxel_size), int(abs(self.pc_range[1] - self.pc_range[4]) / self.coarse_voxel_size), int(abs(self.pc_range[2] - self.pc_range[5]) / self.coarse_voxel_size))
        binary_canvas = torch.zeros(1, canvs_size[2], canvs_size[1], canvs_size[0], dtype=coors.dtype, device=coors.device)

        binary_canvas[:, coors[:,1], coors[:,2], coors[:,3]] = 1

        return binary_canvas
        
        
    def forward(self, inputs):
        cfg = self.cfg
        # we predict points and associated features in 3d space directly
        # we do not use unprojection, so as camera intrinsics
        outputs = {}
        # TODO: Add multive image and voxel predictor 
        self.encoder(inputs, outputs) # (B, N, 3), (B, N, C), (B, N, 3)

        if cfg.model.gaussian_rendering:
            self.process_gt_poses(inputs, outputs)
        # First, Voxelization and decode pre-batch data here
        # get pre-batch point cloud here!
        if self.training:
            # gt_points = self.get_projected_points(inputs, outputs)
            gt_points = self.get_projected_points_nomask(inputs, outputs)
            gt_points_source = self.get_projected_points_source_view(inputs, outputs)
            gt_points_source_nomask = self.get_projected_points_source_view_nomask(inputs, outputs)
        else:
            gt_points = None
        all_batch_outputs, binary_logits, binary_voxel = [], [], []
        padding_number = 0

        for b in range(cfg.data_loader.batch_size):

            pts3d_origin, pts_enc_feat, pts_rgb = outputs[('pts3d', 0)], outputs[('pts_feat', 0)], outputs[('pts_rgb', 0)]

            # xyz = pts3d_origin[0].cpu().numpy() 
            # storePly("/mnt/ziyuxiao/code/GaussianAnything/output/origin.ply", xyz, np.zeros(xyz.shape, dtype=np.uint8))
            # xyz = gt_points_source[0].cpu().numpy() 
            # storePly("/mnt/ziyuxiao/code/GaussianAnything/output/gt.ply", xyz, np.zeros(xyz.shape, dtype=np.uint8))
            # xyz = gt_points_source_nomask[0].cpu().numpy() 
            # storePly("/mnt/ziyuxiao/code/GaussianAnything/output/gt_nomask.ply", xyz, np.zeros(xyz.shape, dtype=np.uint8))

            features, num_points, coors, voxel_centers = prepare_hard_vfe_inputs_scatter_fast(gt_points_source_nomask[b], pts_enc_feat[b], pts_rgb[b], voxel_size=self.voxel_size, point_cloud_range=self.pc_range)
            voxels_features = self.vfe(features, num_points, coors)
            
            # TODO: We need a corase voxel predictor
            batch_binary_logits, _ = self.vox_pred(voxels_features, coors)
            # get gt corase voxel here
            if self.training:
                batch_binary_voxel = self.get_binary_voxels(gt_points[b]).float()
            else:
                batch_binary_voxel = torch.zeros_like(batch_binary_logits)

            padded_coors, padded_voxel_centers = self.padding_voxel_maxpooling(coors, (batch_binary_logits.sigmoid() > 0.5).float())
            padding_number += padded_coors.shape[0]
            padding_tokens = self.mask_token.repeat(padded_coors.shape[0], 1)
            voxels_features = torch.cat([voxels_features, padding_tokens], dim=0)
            voxel_centers = torch.cat([voxel_centers, padded_voxel_centers], dim=0)

            voxel_centers = voxel_centers.unsqueeze(0)
            voxels_features = voxels_features.unsqueeze(0)
            # normalize points in each batch
            mean = voxel_centers.mean(dim=1, keepdim=True) # (B, 1, 3)
            z_median = torch.median(voxel_centers[:, :, 2:], dim=1, keepdim=True)[0] # (B, 1)
            pts3d = (voxel_centers - mean) / (z_median + 1e-6) # (B, N, 3)

            pts3d, pts_feat = self.decoder_3d(pts3d, voxels_features)

            pts3d = torch.cat(pts3d, dim=1)
            pts3d = pts3d * (z_median + 1e-6) + mean # (B, N, 3)
            
            output_batch = self.decoder_gs(pts_feat)
            
            offset = output_batch["gauss_offset"]
            offset = rearrange(offset, "b s c n -> b (s n) c", s=self.cfg.model.gaussians_per_pixel)
            pts3d = pts3d + offset
            
            pts3d_reshape = rearrange(pts3d, "b (s n) c -> b s c n", s=cfg.model.gaussians_per_pixel)
            output_batch["gauss_means"] = pts3d_reshape
            
            all_batch_outputs.append(output_batch)
            binary_voxel.append(batch_binary_voxel)
            binary_logits.append(batch_binary_logits)
            
        # save binary_voxel and binary_logits
        binary_logits, binary_voxel = torch.cat(binary_logits, dim=0), torch.cat(binary_voxel, dim=0)
        
        all_batch_outputs = self.padding_dummy_gaussians(all_batch_outputs)
        
        for key, value in all_batch_outputs.items():
            outputs[key] = value
            
        
        outputs["binary_logits"], outputs["binary_voxel"] = binary_logits, binary_voxel
        outputs["padding_number"] = padding_number / binary_logits.shape[0]


        if cfg.model.gaussian_rendering:
            # self.process_gt_poses(inputs, outputs)
            self.render_images(inputs, outputs)

        return outputs
    
    def padding_dummy_gaussians(self, outputs):
        # This func is using for padding dummy gaussian for batchify the rendering process
        # we padd a very far away gaussians with 0 opacity, try not to affect the rendering process, although it might be little slow
        gaussian_num = []
        for output in outputs:
            dtype, device = output["gauss_scaling"].dtype, output["gauss_scaling"].device
            gaussian_num.append(output["gauss_scaling"].shape[-1])
            
        max_num_gaussians = max(gaussian_num)
        batchify_output = collections.defaultdict(list)
        for output in outputs:
            if output["gauss_scaling"].shape[-1] == max_num_gaussians:
                for key, value in output.items():
                    batchify_output[key].append(value)
                continue
            for key, value in output.items():
                # for opacity, we should padding gaussians 

                padding_num = abs(output["gauss_scaling"].shape[-1] - max_num_gaussians)
                if key == 'gauss_opacity':
                    padding_element = torch.zeros((1, 1, 1, 1), dtype=dtype, device=device).repeat(1, 1, 1, padding_num)
                    batchify_output['gauss_opacity'].append(torch.cat([output['gauss_opacity'], padding_element], dim=-1))
                elif key == 'gauss_scaling':
                    padding_element = torch.zeros((1, 1, 3, 1), dtype=dtype, device=device).repeat(1, 1, 1, padding_num)
                    batchify_output['gauss_scaling'].append(torch.cat([output['gauss_scaling'], padding_element], dim=-1))
                elif key == 'gauss_rotation':
                    padding_element = torch.zeros((1, 1, 4, 1), dtype=dtype, device=device).repeat(1, 1, 1, padding_num)
                    batchify_output['gauss_rotation'].append(torch.cat([output['gauss_rotation'], padding_element], dim=-1))
                elif key == 'gauss_features_dc':
                    feat_dim = output['gauss_features_dc'].shape[2]
                    padding_element = torch.zeros((1, 1, feat_dim, 1), dtype=dtype, device=device).repeat(1, 1, 1, padding_num)
                    batchify_output['gauss_features_dc'].append(torch.cat([output['gauss_features_dc'], padding_element], dim=-1))
                elif key == 'gauss_offset':
                    padding_element = torch.zeros((1, 1, 3, 1), dtype=dtype, device=device).repeat(1, 1, 1, padding_num)
                    batchify_output['gauss_offset'].append(torch.cat([output['gauss_offset'], padding_element], dim=-1))
                elif key == 'gauss_means':
                    padding_element = torch.ones((1, 1, 3, 1), dtype=dtype, device=device).repeat(1, 1, 1, padding_num) * -1000000
                    batchify_output['gauss_means'].append(torch.cat([output['gauss_means'], padding_element], dim=-1))
                    
        for key, value in batchify_output.items():
            batchify_output[key] = torch.cat(value, dim=0)
            
        return batchify_output