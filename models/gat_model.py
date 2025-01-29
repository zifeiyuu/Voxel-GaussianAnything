import torch
import torch.nn.functional as F
import logging
import time
import torch.nn as nn

from pathlib import Path
from einops import rearrange
import collections
import copy

from .encoder.moge_encoder import MoGe_MVEncoder
from .encoder.voxel_encoder import prepare_hard_vfe_inputs_scatter_fast, prepare_voxel_features_scatter_mean, HardVFE, compute_voxel_coors_and_centers
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
        self.voxel_size_factor = cfg.model.coarse_voxel_size // cfg.model.voxel_size
        # self.vfe = HardVFE(in_channels=cfg.model.backbone.pts_feat_dim+3+3, feat_channels=[cfg.model.voxel_feat_dim, cfg.model.voxel_feat_dim], voxel_size=(self.voxel_size, self.voxel_size, self.voxel_size), point_cloud_range=self.pc_range)
        # self.parameters_to_train += [{"params": self.vfe.parameters()}]

        if self.binary_predictor:
            self.vox_pred = VoxPredictor(cfg)
            self.parameters_to_train += self.vox_pred.get_parameter_groups()
        
        self.coarse_canvs_size = (
            int(abs(self.pc_range[0] - self.pc_range[3]) / self.coarse_voxel_size),
            int(abs(self.pc_range[1] - self.pc_range[4]) / self.coarse_voxel_size),
            int(abs(self.pc_range[2] - self.pc_range[5]) / self.coarse_voxel_size)
        )

        kw = dict(copy.deepcopy(cfg.model.unet3d))
        self.unet3d = Modified3DUnet(**kw)
        self.parameters_to_train += [{"params": list(self.unet3d.parameters())}]

        if not self.pre_train_flag:
            self.decoder_3d = PointTransformerDecoder(cfg)
            self.parameters_to_train += self.decoder_3d.get_parameter_groups()
            head_in_dim = self.decoder_3d.transformer.out_dim

            self.decoder_gs = LinearHead(cfg, head_in_dim, xyz_scale=cfg.model.xyz_scale, xyz_bias=cfg.model.xyz_bias, predict_offset=cfg.model.predict_offset)
            self.parameters_to_train += self.decoder_gs.get_parameter_groups()

            # Freeze pretrained parameters
            if cfg.train.freeze_pretrain:
                for name, param in self.named_parameters():
                    if "vfe" in name or "vox_pred" in name:
                        param.requires_grad = False 

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
    
    def padding_voxel_feature(self, src_coors, src_feat, pred_binary_canvas, copy_feature=False):
        # WARNING! This function only take bs=1 input, i.e., the first dim of coords should always be zero
        assert src_coors[..., 0].sum() == 0
        canvs_size = (
            int(abs(self.pc_range[0] - self.pc_range[3]) / self.coarse_voxel_size),
            int(abs(self.pc_range[1] - self.pc_range[4]) / self.coarse_voxel_size),
            int(abs(self.pc_range[2] - self.pc_range[5]) / self.coarse_voxel_size)
        )
        binary_canvas = torch.zeros(1, canvs_size[2], canvs_size[1], canvs_size[0], dtype=src_coors.dtype, device=src_coors.device)
        binary_canvas[:, src_coors[:, 1], src_coors[:, 2], src_coors[:, 3]] = 1

        feat_canvas = torch.zeros(1, self.cfg.model.voxel_feat_dim, pred_binary_canvas.shape[1], pred_binary_canvas.shape[2], pred_binary_canvas.shape[3], dtype=src_feat.dtype, device=src_feat.device)
        feat_canvas[src_coors[:, 0], :, src_coors[:, 1], src_coors[:, 2], src_coors[:, 3]] = src_feat
        rest_binary_canvas = (pred_binary_canvas - binary_canvas).clamp(min=0)
        rest_coors = torch.nonzero(rest_binary_canvas)
        all_coors = torch.cat([src_coors, rest_coors], dim=0)

        if copy_feature:
            # Build KDTree and query nearest neighbors
            occupied_coors_np = src_coors[:, 1:].cpu().numpy()  # Shape: [num_occupied, 3]
            non_occupied_coors_np = rest_coors[:, 1:].cpu().numpy()  # Shape: [num_non_occupied, 3]
            kd_tree = KDTree(occupied_coors_np)
            distances, nearest_indices = kd_tree.query(non_occupied_coors_np, k=1)
            nearest_indices = torch.tensor(nearest_indices, device=src_coors.device)  # Shape: [num_non_occupied]
            nearest_features = feat_canvas[
                src_coors[nearest_indices, 0], :,  
                src_coors[nearest_indices, 1], 
                src_coors[nearest_indices, 2], 
                src_coors[nearest_indices, 3]  
            ]  # Shape: [num_non_occupied, feature_dim]
            feat_canvas[
                rest_coors[:, 0], :,  
                rest_coors[:, 1], 
                rest_coors[:, 2],
                rest_coors[:, 3]  
            ] = nearest_features

        all_feats = feat_canvas[all_coors[:, 0], :, all_coors[:, 1], all_coors[:, 2], all_coors[:, 3]] # Shape: [M, 64]

        # Compute all voxel centers
        grid_min = self.pc_range[:3]
        all_voxel_centers = torch.stack([
            all_coors[:, -1].float() * self.coarse_voxel_size + grid_min[0] + self.coarse_voxel_size / 2,
            all_coors[:, -2].float() * self.coarse_voxel_size + grid_min[1] + self.coarse_voxel_size / 2,
            all_coors[:, -3].float() * self.coarse_voxel_size + grid_min[2] + self.coarse_voxel_size / 2,
        ], dim=1)  

        return all_coors, all_voxel_centers, all_feats, rest_binary_canvas
    

    def get_projected_points(self, inputs, outputs, pretrain=False):
        depth_error = False
        eps, max_depth = 1e-3, 20
        # we project points from target view and novel view
        pts3d_batch = []
        pts3d_dict_batch = []
        using_frames = self.using_frames
        if pretrain:
            B = len(inputs[('depth_sparse', 0)])
        else:
            B = outputs[('depth_pred', 0)].shape[0]

        for batch_idx in range(B):    # outputs[('depth_pred', 0)]    inputs[('depth_sparse', 0)]
            pts3d = []
            pts3d_dict = {}
            for frameid in using_frames:
                device = inputs[('K_tgt', frameid)][batch_idx].device
                if pretrain:
                    depth = inputs[('depth_sparse', frameid)][batch_idx].to(device)
                else:
                    depth = outputs[('depth_pred', frameid)][batch_idx].squeeze(0).to(device)
                K = inputs[('K_tgt', frameid)][batch_idx]
                c2w = outputs[('cam_T_cam', frameid, 0)][batch_idx] if frameid != 0 else None
                
                mask = torch.logical_and((depth > eps), (depth < max_depth)).bool()
                # depth[~mask] = max_depth
                if depth[mask].numel() > 0:
                    depth[~mask] = depth[mask].max()
                else:
                    depth_error = True
                    if pretrain and ('depth_pred', frameid) in outputs.keys() and torch.logical_and((outputs[('depth_pred', frameid)] > eps), (outputs[('depth_pred', frameid)] < max_depth)).bool().sum() > 0:
                        depth = outputs[('depth_pred', frameid)][batch_idx].squeeze(0).to(device)
                    else:
                        print("depth map error, use default depth: max depth")
                        depth[~mask] = max_depth / 4 # Or some other default value

                _pts3d, _ = depthmap_to_absolute_camera_coordinates_torch(depth, K, c2w)
                # pts3d.append(_pts3d[mask])
                pts3d.append(_pts3d.flatten(0, 1))
                
                pts3d_dict[frameid] = _pts3d.flatten(0, 1)
                
            pts3d = torch.cat(pts3d)
            
            pts3d_batch.append(pts3d)
            pts3d_dict_batch.append(pts3d_dict)
        return pts3d_batch, pts3d_dict_batch, depth_error
    
    
    def get_binary_voxels(self, points):
        coors, voxel_centers = compute_voxel_coors_and_centers(points, voxel_size=self.coarse_voxel_size, point_cloud_range=self.pc_range)
        
        canvs_size = (int(abs(self.pc_range[0] - self.pc_range[3]) / self.coarse_voxel_size), int(abs(self.pc_range[1] - self.pc_range[4]) / self.coarse_voxel_size), int(abs(self.pc_range[2] - self.pc_range[5]) / self.coarse_voxel_size))
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
        if self.pre_train_flag:
            output = self.forward_voxpred(inputs)
        else:
            output = self.forward_gsm(inputs)
            
        return output
        
    def forward_voxpred(self, inputs):
        cfg = self.cfg
        B, C, H, W = inputs["color_aug", 0, 0].shape
        # we predict points and associated features in 3d space directly
        # we do not use unprojection, so as camera intrinsics
        outputs = {}
        # TODO: Add multive image and voxel predictor 
        self.encoder(inputs, outputs) # (B, N, 3), (B, N, C), (B, N, 3)

        if cfg.model.gaussian_rendering:
            self.process_gt_poses(inputs, outputs, pretrain=True)
        # First, Voxelization and decode pre-batch data here
        # get pre-batch point cloud here!
        gt_points, gt_points_dict, depth_error = self.get_projected_points(inputs, outputs, pretrain=True)
        outputs["gt_points"] = gt_points
        outputs["error"] = depth_error
        coors_list, voxels_features_list = [], []
        binary_logits, binary_voxel, rest_binary_voxel_list = [], [], []
        
        for b in range(cfg.data_loader.batch_size):
            pts_enc_feat = outputs[('pts_feat', 0)][b]
            pts_rgb = outputs[('pts_rgb', 0)][b]

            # ONLY SRC VIEW VOXELS HERE
            voxels_features, coors, voxel_centers = prepare_voxel_features_scatter_mean(gt_points_dict[b][0], pts_enc_feat, pts_rgb, voxel_size=self.voxel_size, point_cloud_range=self.pc_range)
            # features, num_points, coors, voxel_centers = prepare_hard_vfe_inputs_scatter_fast(gt_points_dict[b][0], pts_enc_feat, pts_rgb, voxel_size=self.voxel_size, point_cloud_range=self.pc_range)
            # voxels_features = self.vfe(features, num_points, coors)      

            if self.binary_predictor:
                # # SRC + NOVEL VIEW VOXELS(PREDICTED)
                batch_binary_logits, _, batch_pred_feat = self.vox_pred(voxels_features, coors, max_pooling=True)  # batch_pred_feat (B, Z, Y, X, 64)
                # get gt corase voxel here
                batch_binary_voxel, batch_gt_voxel_centers = self.get_binary_voxels(gt_points[b])
                batch_binary_voxel = batch_binary_voxel.float()
                batch_gt_voxel_centers = batch_gt_voxel_centers.float()
                gt_coors = torch.nonzero(batch_binary_voxel == 1, as_tuple=False) # Shape: [M, 4]

                padding_coors, padding_voxel_centers, padding_features, rest_padding_binary_voxel = self.padding_voxel_feature(coors, voxels_features, (batch_binary_logits.sigmoid() > 0.5).float(), copy_feature=False)   # batch_binary_voxel  (batch_binary_logits.sigmoid() > 0.5).float()
                
                coors = padding_coors
                voxels_features = padding_features

                binary_voxel.append(batch_binary_voxel)
                binary_logits.append(batch_binary_logits)
                rest_binary_voxel_list.append(rest_padding_binary_voxel)

            coors_list.append(coors[:, [3,2,1]].to(torch.int64))   #zyx to xyz
            voxels_features_list.append(voxels_features)

        K_6d = self.intrinsic_matrix_to_6d(inputs[("K_src", 0)], H, W)
        img_features = rearrange(outputs[('pts_feat', 0)].unsqueeze(1), "b n (h w) c -> b n c h w", h=H, w=W)
        #3D UNET
        position_list, opacity_list, scaling_list, rotation_list, feat_dc_list = self.unet3d(coors_list, [self.voxel_size]*3, self.pc_range[0:3], voxels_features_list, img_features, outputs[("cam_T_cam", 0, 0)].unsqueeze(1), K_6d)
        outputs["gauss_means"] = position_list
        outputs["gauss_opacity"] = opacity_list
        outputs["gauss_scaling"] = scaling_list
        outputs["gauss_rotation"] = rotation_list
        outputs["gauss_features_dc"] = feat_dc_list

        if self.binary_predictor:
            # save binary_voxel and binary_logits
            binary_logits, binary_voxel, rest_binary_voxel = torch.cat(binary_logits, dim=0), torch.cat(binary_voxel, dim=0), torch.cat(rest_binary_voxel_list, dim=0)
            outputs["binary_logits"], outputs["binary_voxel"], outputs["rest_binary_voxel"] = binary_logits, binary_voxel, rest_binary_voxel

        if cfg.model.gaussian_rendering:
            self.render_images(inputs, outputs)

        return outputs

    def forward_gsm(self, inputs):
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
            gt_points, _ = self.get_projected_points(inputs, outputs)
        else:
            gt_points = None

        all_batch_outputs, binary_logits, binary_voxel = [], [], []
        padding_number = 0

        for b in range(cfg.data_loader.batch_size):

            pts3d_moge, pts_enc_feat, pts_rgb = outputs[('pts3d', 0)], outputs[('pts_feat', 0)], outputs[('pts_rgb', 0)]

            features, num_points, coors, voxel_centers = prepare_hard_vfe_inputs_scatter_fast(pts3d_moge[b], pts_enc_feat[b], pts_rgb[b], voxel_size=self.voxel_size, point_cloud_range=self.pc_range)
            voxels_features = self.vfe(features, num_points, coors)
            
            # TODO: We need a corase voxel predictor
            batch_binary_logits, _, batch_pred_feat = self.vox_pred(voxels_features, coors)  # batch_pred_feat (B, Z, Y, X, 64)
            # get gt corase voxel here
            if self.training:
                batch_binary_voxel, _ = self.get_binary_voxels(gt_points[b])
            else:
                batch_binary_voxel = torch.zeros_like(batch_binary_logits)
            batch_binary_voxel = batch_binary_voxel.float()

            ## 用batch_binary_logits mask 补全 ##
            ## padding_all_coors could not be directly used since voxel size doesn't match
            ## 这里还可加一步 coarse-to-fine??
            coarse_src_coors, _ = compute_voxel_coors_and_centers(pts3d_moge[b], voxel_size=self.coarse_voxel_size, point_cloud_range=self.pc_range)
            padding_all_coors, padding_all_voxel_centers, padding_all_feats = self.padding_voxel_maxpooling_coarse(coarse_src_coors, (batch_binary_logits.sigmoid() > 0.5).float(), batch_pred_feat)
            padding_number += (padding_all_coors.shape[0] - coors.shape[0])

            ### 用 <src fine> combine <predicted coarse> 为了render时候增加voxel数量？
            voxel_centers = torch.cat([voxel_centers, padding_all_voxel_centers])
            voxels_features = torch.cat([voxels_features, padding_all_feats])

            # PADDING 1: 用src fine + coarse prediction 先出一份gaussian
            output_batch_pretrain = self.decoder_gs_padding([voxels_features])
            batch_voxel_centers = voxel_centers.unsqueeze(0)
            pts3d_reshape = rearrange(batch_voxel_centers, "b (s n) c -> b s c n", s=cfg.model.gaussians_per_pixel)
            output_batch_pretrain["gauss_means"] = pts3d_reshape

            ## PADDING 2: combined <src view fine xyz/feature> and <coarse xyz/feature> go through point transformer##
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
            if cfg.model.predict_offset:
                offset = output_batch["gauss_offset"]
                offset = rearrange(offset, "b s c n -> b (s n) c", s=self.cfg.model.gaussians_per_pixel)
                pts3d = pts3d + offset
            pts3d_reshape = rearrange(pts3d, "b (s n) c -> b s c n", s=cfg.model.gaussians_per_pixel)
            output_batch["gauss_means"] = pts3d_reshape

            for key, value in output_batch.items():
                if key == "gauss_offset":
                    output_batch[key] = torch.cat([value, torch.zeros_like(output_batch_pretrain["gauss_means"])], dim=-1)
                else:
                    output_batch[key] = torch.cat([value, output_batch_pretrain[key]], dim=-1)
            
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