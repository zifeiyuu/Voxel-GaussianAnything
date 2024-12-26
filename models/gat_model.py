import torch
import torch.nn.functional as F
import logging
import time
import torch.nn as nn

from pathlib import Path
from einops import rearrange
import collections

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
        if "moge" in cfg.model.backbone.name:
            self.encoder = MoGe_Encoder(cfg)
            
        self.parameters_to_train += self.encoder.get_parameter_groups()
        head_in_dim = [cfg.model.backbone.pts_feat_dim]

        self.voxel_size, self.pc_range = cfg.model.voxel_size, cfg.model.pc_range
        self.vfe = HardVFE(in_channels=cfg.model.backbone.pts_feat_dim+3+3, feat_channels=[64, 64], voxel_size=(self.voxel_size, self.voxel_size, self.voxel_size), point_cloud_range=self.pc_range)
        self.parameters_to_train += [{"params": self.vfe.parameters()}]

        self.decoder_3d = PointTransformerDecoder(cfg)
        self.parameters_to_train += self.decoder_3d.get_parameter_groups()
        head_in_dim = self.decoder_3d.transformer.out_dim

        self.decoder_gs = LinearHead(cfg, head_in_dim, xyz_scale=cfg.model.xyz_scale, xyz_bias=cfg.model.xyz_bias)


        self.parameters_to_train += self.decoder_gs.get_parameter_groups()

        self.use_checkpoint = False
        self.use_reentrant = False
        
    
        

    def forward(self, inputs):
        cfg = self.cfg
        # we predict points and associated features in 3d space directly
        # we do not use unprojection, so as camera intrinsics

        pts3d_origin, pts_enc_feat, pts_rgb, pts_depth_origin, padding_select = self.encoder(inputs) # (B, N, 3), (B, N, C), (B, N, 3)

        # First, Voxelization and decode pre-batch data here
        outputs = []
        for b in range(pts3d_origin.shape[0]):

            features, num_points, coors, voxel_centers = prepare_hard_vfe_inputs_scatter_fast(pts3d_origin[b], pts_enc_feat[b], pts_rgb[b], voxel_size=self.voxel_size, point_cloud_range=self.pc_range)
            voxels_features = self.vfe(features, num_points, coors)
            
            # TODO: We need a corase voxel predictor
            
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
            
            outputs.append(output_batch)

        outputs = self.padding_dummy_gaussians(outputs)
            
        outputs[("depth", 0)] = pts_depth_origin

        if cfg.model.gaussian_rendering:
            self.process_gt_poses(inputs, outputs)
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