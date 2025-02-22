import torch
import torch.nn.functional as F
import logging
import time
import torch.nn as nn

from pathlib import Path
from einops import rearrange
import collections
import copy
import cv2
from .encoder.moge_encoder import MoGe_MVEncoder
from .encoder.voxel_encoder import prepare_voxel_features_scatter_mean, HardVFE, compute_voxel_coors_and_centers
from .decoder.gauss_util import focal2fov, getProjectionMatrix, K_to_NDC_pp, render_predicted
from .base_model import BaseModel
from .heads.gat_head import LinearHead
from misc.util import add_source_frame_id
from misc.depth import estimate_depth_scale, estimate_depth_scale_ransac, depthmap_to_absolute_camera_coordinates_torch
from misc.visualise_3d import storePly # for debugging
import numpy as np
from torch_scatter import scatter_max
from collections import deque
from scipy.spatial import KDTree

from .decoder.pointcloud_decoder import PointTransformerDecoder
from .decoder.voxel_predictor import VoxPredictor
from .decoder.unet_3d import Modified3DUnet

from IPython import embed
from pdb import set_trace

from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_mean
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast

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
        self.pre_train_flag = cfg.train.pretrain
        self.using_frames = [0] + cfg.model.gauss_novel_frames
        self.binary_predictor = cfg.model.binary_predictor
        self.parameters_to_train = []

        # define the model
        if "moge" in cfg.model.backbone.name:
            self.encoder = MoGe_MVEncoder(cfg)
            self.parameters_to_train += self.encoder.get_parameter_groups()
        head_in_dim = [cfg.model.backbone.pts_feat_dim]

        self.voxel_size, self.pc_range, self.coarse_voxel_size = cfg.model.voxel_size, cfg.model.pc_range, cfg.model.coarse_voxel_size
        self.voxel_size_factor = cfg.model.coarse_voxel_size / cfg.model.voxel_size
        # self.vfe = HardVFE(in_channels=cfg.model.backbone.pts_feat_dim+3+3, feat_channels=[cfg.model.voxel_feat_dim, cfg.model.voxel_feat_dim], voxel_size=(self.voxel_size, self.voxel_size, self.voxel_size), point_cloud_range=self.pc_range)
        # self.parameters_to_train += [{"params": self.vfe.parameters()}]

        if self.binary_predictor:
            self.vox_pred = VoxPredictor(cfg)
            # self.vox_pred.to(dtype=torch.bfloat16)
            self.parameters_to_train += self.vox_pred.get_parameter_groups()
            

        kw = dict(copy.deepcopy(cfg.model.unet3d))
        self.unet3d = Modified3DUnet(**kw)
        self.parameters_to_train += [{"params": list(self.unet3d.parameters())}]
        
    def padding_voxel_feature(self, src_coors, pred_confidence):
        # WARNING! This function only take bs=1 input, i.e., the first dim of coords should always be zero
        assert src_coors[..., 0].sum() == 0

        pred_binary_canvas = (pred_confidence.sigmoid() > 0.5).float()

        upsampling_factor = self.coarse_voxel_size / self.voxel_size 

        # Compute coarse canvas size
        coarse_canvas_size = (
            int(abs(self.pc_range[0] - self.pc_range[3]) / self.coarse_voxel_size),
            int(abs(self.pc_range[1] - self.pc_range[4]) / self.coarse_voxel_size),
            int(abs(self.pc_range[2] - self.pc_range[5]) / self.coarse_voxel_size)
        )
        coarse_binary_canvas = torch.zeros(1, coarse_canvas_size[2], coarse_canvas_size[1], coarse_canvas_size[0], 
                                        dtype=src_coors.dtype, device=src_coors.device)

        # Convert fine voxel indices to coarse voxel indices
        coarse_src_coors = src_coors.clone()
        coarse_src_coors[:, 1:] = (coarse_src_coors[:, 1:] / upsampling_factor).round().long()
        max_x, max_y, max_z = coarse_binary_canvas.shape[-1], coarse_binary_canvas.shape[-2], coarse_binary_canvas.shape[-3]
        # Ensure indices are within valid bounds
        coarse_src_coors[:, 1] = coarse_src_coors[:, 1].clamp(min=0, max=max_z - 1)
        coarse_src_coors[:, 2] = coarse_src_coors[:, 2].clamp(min=0, max=max_y - 1)
        coarse_src_coors[:, 3] = coarse_src_coors[:, 3].clamp(min=0, max=max_x - 1)
        coarse_binary_canvas[:, coarse_src_coors[:, 1], coarse_src_coors[:, 2], coarse_src_coors[:, 3]] = 1

        rest_binary_canvas = (pred_binary_canvas - coarse_binary_canvas).clamp(min=0)
        coarse_rest_coors = torch.nonzero(rest_binary_canvas)

        if coarse_rest_coors.numel() == 0:  
            return coarse_rest_coors, torch.ones(0, 1, dtype=src_coors.dtype, device=src_coors.device), torch.ones(0, 3, dtype=src_coors.dtype, device=src_coors.device) # , torch.ones(1, 3, dtype=src_coors.dtype, device=src_coors.device)

        # Convert coarse coors to fine

        fine_offsets = torch.tensor([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1],  [0, 0, -1], 
            [0, 0, 0]  
        ], device=coarse_rest_coors.device) * (upsampling_factor // 2)

        # Use 8 corners + center
        # fine_offsets = torch.tensor([
        #     [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        #     [-1, -1, 1],  [1, -1, 1],  [-1, 1, 1],  [1, 1, 1], 
        #     [0, 0, 0]  
        # ], device=coarse_rest_coors.device) * (upsampling_factor // 2)
        # fine_offsets = torch.tensor([[0, 0, 0]], device=rest_coors.device)

        rest_coors, rest_fine_confidence = self.upsample_voxel_coords(coarse_rest_coors, src_coors, fine_offsets, pred_confidence)   # input rest coors is coarse, src_coors is fine

        grid_min = self.pc_range[:3]
        rest_voxel_centers = torch.stack([
            rest_coors[:, -1].float() * self.voxel_size + grid_min[0] + self.voxel_size / 2,
            rest_coors[:, -2].float() * self.voxel_size + grid_min[1] + self.voxel_size / 2,
            rest_coors[:, -3].float() * self.voxel_size + grid_min[2] + self.voxel_size / 2,
        ], dim=1) 

        return rest_coors, rest_fine_confidence, rest_voxel_centers
    
    def upsample_voxel_coords(self, coarse_rest_coors, src_coors, fine_offsets, pred_confidence):

        upsampling_factor = self.coarse_voxel_size / self.voxel_size

        fine_canvas_size = (
            int(abs(self.pc_range[0] - self.pc_range[3]) / self.voxel_size),
            int(abs(self.pc_range[1] - self.pc_range[4]) / self.voxel_size),
            int(abs(self.pc_range[2] - self.pc_range[5]) / self.voxel_size)
        )
        fine_binary_canvas = torch.zeros(1, fine_canvas_size[2], fine_canvas_size[1], fine_canvas_size[0], dtype=coarse_rest_coors.dtype, device=coarse_rest_coors.device)

        fine_offset_x = fine_offsets[:, 0]
        fine_offset_y = fine_offsets[:, 1]
        fine_offset_z = fine_offsets[:, 2]

        max_x, max_y, max_z = fine_binary_canvas.shape[-1], fine_binary_canvas.shape[-2], fine_binary_canvas.shape[-3]
        # Repeat coarse voxel indices for each fine voxel inside it
        fine_rest_coors = coarse_rest_coors.repeat_interleave(fine_offset_x.shape[0], dim=0)
        fine_rest_coors[:, -1] = (fine_rest_coors[:, -1] * upsampling_factor).round() + fine_offset_x.repeat(coarse_rest_coors.shape[0])
        fine_rest_coors[:, -2] = (fine_rest_coors[:, -2] * upsampling_factor).round() + fine_offset_y.repeat(coarse_rest_coors.shape[0])
        fine_rest_coors[:, -3] = (fine_rest_coors[:, -3] * upsampling_factor).round() + fine_offset_z.repeat(coarse_rest_coors.shape[0])

        # Clamp coordinates to be within valid bounds
        fine_rest_coors[:, 1] = fine_rest_coors[:, 1].clamp(min=0, max=max_z - 1)
        fine_rest_coors[:, 2] = fine_rest_coors[:, 2].clamp(min=0, max=max_y - 1)
        fine_rest_coors[:, 3] = fine_rest_coors[:, 3].clamp(min=0, max=max_x - 1)

        coarse_confidence = pred_confidence[coarse_rest_coors[:, 0], coarse_rest_coors[:, 1], coarse_rest_coors[:, 2], coarse_rest_coors[:, 3]]
        # Repeat the coarse confidence values to match fine coordinates
        fine_confidence = coarse_confidence.repeat_interleave(fine_offsets.shape[0], dim=0)

        # Fill fine binary canvas
        rest_binary_canvas = fine_binary_canvas.float()
        # rest_binary_canvas = rest_binary_canvas.to(torch.bfloat16)
        rest_binary_canvas[:, fine_rest_coors[:, 1], fine_rest_coors[:, 2], fine_rest_coors[:, 3]] = fine_confidence
        fine_binary_canvas[:, src_coors[:, 1], src_coors[:, 2], src_coors[:, 3]] = 1
        # ensure no overlap
        rest_binary_canvas = rest_binary_canvas * (fine_binary_canvas == 0)
        rest_coors = torch.nonzero(rest_binary_canvas)
        fine_confidence = rest_binary_canvas[:, rest_coors[:, 1], rest_coors[:, 2], rest_coors[:, 3]].view(-1, 1)

        return rest_coors, fine_confidence #rest_binary_canvas, (rest_binary_canvas + fine_binary_canvas).clamp(max=1)
    
    def get_projected_points(self, inputs, outputs):
        depth_error = False
        # we project points from target view and novel view
        pts3d_batch = []
        using_frames = self.using_frames

        B = len(inputs[('depth_sparse', 0)])
        src_scale = outputs[('depth_scale', 0)]

        for batch_idx in range(B):    # outputs[('depth_pred', 0)]    inputs[('depth_sparse', 0)]
            pts3d = []
            for frameid in using_frames:
                tgt_scale = outputs[('depth_scale', frameid)]
                device = inputs[('K_tgt', frameid)][batch_idx].device

                depth = outputs[('depth_pred', frameid)][batch_idx].squeeze().to(device) / tgt_scale[batch_idx] * src_scale[batch_idx]
                K = inputs[('K_tgt', frameid)][batch_idx]
                c2w = outputs[('cam_T_cam', frameid, 0)][batch_idx] if frameid != 0 else None
                # breakpoint()
                _pts3d, _ = depthmap_to_absolute_camera_coordinates_torch(depth, K, c2w)
                pts3d.append(_pts3d.flatten(0, 1))
            pts3d_batch.append(torch.cat(pts3d))

        return pts3d_batch, depth_error
    
    
    def get_binary_voxels(self, points, voxel_size):
        # _, coors, voxel_centers = prepare_voxel_features_scatter_mean(points, torch.zeros_like(points), torch.zeros_like(points), voxel_size=voxel_size, point_cloud_range=self.pc_range)
        coors, voxel_centers = compute_voxel_coors_and_centers(points, voxel_size=voxel_size, point_cloud_range=self.pc_range)
        
        canvs_size = (int(abs(self.pc_range[0] - self.pc_range[3]) / voxel_size), int(abs(self.pc_range[1] - self.pc_range[4]) / voxel_size), int(abs(self.pc_range[2] - self.pc_range[5]) / voxel_size))
        binary_canvas = torch.zeros(1, canvs_size[2], canvs_size[1], canvs_size[0], dtype=coors.dtype, device=coors.device)

        binary_canvas[:, coors[:,1], coors[:,2], coors[:,3]] = 1

        return binary_canvas, voxel_centers
    
    def voxel_max_pooling(self, voxels_features, coors):
        batch = coors[:, 0]
        z = coors[:, 1] // self.voxel_size_factor
        x = coors[:, 2] // self.voxel_size_factor
        y = coors[:, 3] // self.voxel_size_factor
        
        coors_reduced = torch.stack([batch, z, x, y], dim=-1)  # [N, 4]

        unique_coors, inverse_indices = torch.unique(coors_reduced, return_inverse=True, dim=0)

        new_voxels_features, _ = scatter_max(voxels_features, inverse_indices, dim=0)
        new_coors = unique_coors
        
        return new_voxels_features, new_coors
    
    def sinusoidal_encoding(self, grid_size, embed_dim, device='cpu'):
        D, H, W = grid_size
        z_pos = torch.arange(D, device=device).unsqueeze(-1)
        x_pos = torch.arange(H, device=device).unsqueeze(-1)
        y_pos = torch.arange(W, device=device).unsqueeze(-1)

        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device) * (-torch.log(torch.tensor(10000.0, device=device)) / embed_dim))

        z_embed = torch.cat([torch.sin(z_pos * div_term), torch.cos(z_pos * div_term)], dim=-1)
        x_embed = torch.cat([torch.sin(x_pos * div_term), torch.cos(x_pos * div_term)], dim=-1)
        y_embed = torch.cat([torch.sin(y_pos * div_term), torch.cos(y_pos * div_term)], dim=-1)

        z_embed = z_embed.unsqueeze(1).unsqueeze(2).expand(D, H, W, -1)
        x_embed = x_embed.unsqueeze(0).unsqueeze(2).expand(D, H, W, -1)
        y_embed = y_embed.unsqueeze(0).unsqueeze(1).expand(D, H, W, -1)

        embed = z_embed + x_embed + y_embed

        return embed.permute(3, 0, 1, 2).unsqueeze(0)  # Shape: (1, embed_dim, D, H, W)

    def forward(self, inputs):
        cfg = self.cfg
        B, C, H, W = inputs["color_aug", 0, 0].shape
        # we predict points and associated features in 3d space directly
        # we do not use unprojection, so as camera intrinsics
        outputs = {}
        # TODO: Add multive image and voxel predictor 
        self.encoder(inputs, outputs) # (B, N, 3), (B, N, C), (B, N, 3)
        outputs["error"] = False

        self.process_gt_poses(inputs, outputs, pretrain=self.pre_train_flag)
        # First, Voxelization and decode pre-batch data here
        # get pre-batch point cloud here!
        gt_points, depth_error = self.get_projected_points(inputs, outputs)  ####### scale GT depth scale to MOGE scale????
        outputs["gt_points"] = gt_points
        outputs["error"] = depth_error or outputs["error"]
        coors_list, padding_coors_list, voxels_features_list = [], [], []
        binary_logits, binary_voxel = [], []
        padding_confidence_list = []

        outputs["padding_points"] = []
        outputs["coarse_padding_points"] = []
        
        for b in range(cfg.data_loader.batch_size):
            pts_enc_feat = outputs[('pts_feat', 0)][b]
            pts_rgb = outputs[('pts_rgb', 0)][b]
            pts_xyz = outputs[("pts3d", 0)][b]
            pts_feat = torch.cat([pts_enc_feat, pts_rgb], dim=-1)

            # ONLY SRC VIEW VOXELS HERE
            voxels_features, coors, _ = prepare_voxel_features_scatter_mean(pts_xyz, pts_feat, pts_rgb, voxel_size=self.voxel_size, point_cloud_range=self.pc_range)
            # features, num_points, coors, voxel_centers = prepare_hard_vfe_inputs_scatter_fast(gt_points_dict[b][0], pts_enc_feat, pts_rgb, voxel_size=self.voxel_size, point_cloud_range=self.pc_range)
            # voxels_features = self.vfe(features, num_points, coors)      

            if self.binary_predictor:
                # # SRC + NOVEL VIEW VOXELS(PREDICTED)
                # coarse voxel predicted
                # with autocast(dtype=torch.bfloat16):  
                batch_binary_logits, gt_binary_voxel_single, coor = self.vox_pred(voxels_features, coors, max_pooling=True)  # batch_pred_feat (B, Z, Y, X, 64)
                # get gt corase voxel (fine)
                gt_binary_voxel, _ = self.get_binary_voxels(gt_points[b], voxel_size=self.coarse_voxel_size)
                gt_binary_voxel = gt_binary_voxel.float()
                
                valid_voxels = (batch_binary_logits.sigmoid() > 0.5).float()
                confidence_values = batch_binary_logits[valid_voxels > 0.5]
                confidence_values = confidence_values.view(-1, 1)

                padding_coors, padding_confidence, padding_xyz = self.padding_voxel_feature(coors, batch_binary_logits)   # gt_binary_voxel  (batch_binary_logits.sigmoid() > 0.5).float()

                binary_voxel.append(gt_binary_voxel.float())
                binary_logits.append(batch_binary_logits.float())
                padding_coors_list.append(padding_coors[:, [3,2,1]].to(torch.int64))
                padding_confidence_list.append(padding_confidence)
                outputs["padding_points"].append(padding_xyz)

            coors_list.append(coors[:, [3,2,1]].to(torch.int64))   #zyx to xyz
            voxels_features_list.append(voxels_features)

        ############ avoid memory full
        # max_points_coor = 47000
        # for p in coors_list:
        #     if p.shape[0] > max_points_coor:
        #         outputs["error"] = True
                
        # max_points_pad = 16e4
        # for p in padding_coors_list:
        #     if p.shape[0] > max_points_pad:
        #         outputs["error"] = True

        # if outputs["error"] == True:
        #     for b in range(B):
        #         coors_list[b] = coors_list[b][: 1]
        #         voxels_features_list[b] = voxels_features_list[b][: 1]
        #         padding_coors_list[b] = padding_coors_list[b][: 1]
        #         padding_confidence_list[b] = padding_confidence_list[b][: 1]
        #     print("TO AVOID OUT OF MEMORY, SKIP THIS BATCH")
        #     torch.cuda.empty_cache()
        ###############

        K_6d = self.intrinsic_matrix_to_6d(inputs[("K_src", 0)], H, W)
        img_features = rearrange(torch.cat([outputs[('pts_feat', 0)], outputs[('pts_rgb', 0)]], dim=-1).unsqueeze(1), "b n (h w) c -> b n c h w", h=H, w=W)
        #3D UNET
        # breakpoint()
        position_list, opacity_list, scaling_list, rotation_list, feat_dc_list = self.unet3d(coors_list, padding_coors_list, [self.voxel_size]*3, self.pc_range[0:3], voxels_features_list, img_features, outputs[("cam_T_cam", 0, 0)].unsqueeze(1), K_6d, outputs[("depth_pred", 0)].unsqueeze(1), padding_confidence_list)

        outputs["gauss_means"] = position_list
        outputs["gauss_opacity"] = opacity_list
        outputs["gauss_scaling"] = scaling_list
        outputs["gauss_rotation"] = rotation_list
        outputs["gauss_features_dc"] = feat_dc_list
            
        if self.binary_predictor:
            # save binary_voxel and binary_logits
            # TODO: /home/maoyucheng/code/GaussianAnything/trainer.py: 170 Force set rest voxel to 0, remember to fix
            binary_logits, binary_voxel = torch.cat(binary_logits, dim=0), torch.cat(binary_voxel, dim=0)
            outputs["binary_logits"], outputs["binary_voxel"] = binary_logits, binary_voxel

        if cfg.model.gaussian_rendering:
            self.render_images(inputs, outputs)

        return outputs
    
    def intrinsic_matrix_to_6d(self, K, h, w):
        """
        Convert intrinsic matrix [B, 3, 3] to [B, N, 6] where N=1.
        """
        B = K.shape[0]

        fx = K[:, 0, 0]  # Focal length in x
        fy = K[:, 1, 1]  # Focal length in y
        cx = K[:, 0, 2]  # Principal point x
        cy = K[:, 1, 2]  # Principal point y

        intrinsics_6d = torch.stack([fx, fy, cx, cy, torch.full_like(fx, w), torch.full_like(fy, h)], dim=-1)

        return intrinsics_6d.unsqueeze(1)

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