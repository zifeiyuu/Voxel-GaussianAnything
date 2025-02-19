# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import fvdb
import fvdb.nn as fvnn
from fvdb import GridBatch, JaggedTensor
from fvdb.nn import VDBTensor
from torch_scatter import scatter_mean
from IPython import embed
from pdb import set_trace

class depth_wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, *args):
        return self.module(*args), 0

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, order: str, num_groups: int, kernel_size: int = 3):
        super().__init__()
        for i, char in enumerate(order):
            if char == 'r':
                self.add_module('ReLU', fvnn.ReLU(inplace=True))
            elif char == 's':
                self.add_module('SiLU', fvnn.SiLU(inplace=True))
            elif char == 'c':
                self.add_module('Conv', fvnn.SparseConv3d(
                    in_channels, out_channels, kernel_size, 1, bias='g' not in order))
            elif char == 'g':
                num_channels = in_channels if i < order.index('c') else out_channels
                if num_channels < num_groups:
                    num_groups = 1
                self.add_module('GroupNorm', fvnn.GroupNorm(
                    num_groups=num_groups, num_channels=num_channels, affine=True))
            else:
                raise NotImplementedError

class SparseHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, order, num_groups, enhanced="None"):
        super().__init__()
        self.add_module('SingleConv', ConvBlock(in_channels, in_channels, order, num_groups))
        mid_channels = in_channels
        if out_channels > mid_channels:
            mid_channels = out_channels

        if enhanced == 'three':
            self.add_module('OutConv-1', fvnn.Linear(in_channels, mid_channels))
            self.add_module('ReLU-1', fvnn.LeakyReLU(inplace=True))
            self.add_module('OutConv-2', fvnn.Linear(mid_channels, mid_channels))
            self.add_module('ReLU-2', fvnn.LeakyReLU(inplace=True))
            self.add_module('OutConv', fvnn.Linear(mid_channels, out_channels)) # !: final linear keep name consistent
        elif enhanced == 'upsample':
            self.add_module('upsample', fvnn.UpsamplingNearest(2)) # !: upsample
            self.add_module('OutConv-1', SparseResBlock(in_channels, # ! add back skip connection
                                                        mid_channels,
                                                        order, num_groups, False, None,
                                                        return_feat_depth=False))
            self.add_module('OutConv', fvnn.Linear(mid_channels, out_channels)) # !: final linear keep name consistent
        else:
            self.add_module('OutConv', fvnn.SparseConv3d(in_channels, out_channels, 1, bias=True))

class LinearHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, order, num_groups, enhanced="None"):
        super().__init__()
        mid_channels = in_channels
        if out_channels > mid_channels:
            mid_channels = out_channels
        
        if enhanced == 'three':
            self.add_module('OutConv-1', fvnn.Linear(in_channels, mid_channels))
            self.add_module('ReLU-1', fvnn.LeakyReLU(inplace=True))
            self.add_module('OutConv-2', fvnn.Linear(mid_channels, mid_channels))
            self.add_module('ReLU-2', fvnn.LeakyReLU(inplace=True))
            self.add_module('OutConv', fvnn.Linear(mid_channels, out_channels)) # !: final linear keep name consistent
        elif enhanced == 'original':
            self.add_module('SingleConv', ConvBlock(in_channels, in_channels, order, num_groups))
            self.add_module('OutConv', fvnn.SparseConv3d(in_channels, out_channels, 1, bias=True))

class SparseResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 order: str,
                 num_groups: int,
                 encoder: bool,
                 pooling = None,
                 use_checkpoint: bool = False,
                 return_feat_depth: bool = True
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.return_feat_depth = return_feat_depth

        self.use_pooling = pooling is not None and encoder

        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
            if pooling == 'max':
                self.maxpooling = fvnn.MaxPool(2)
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.conv1 = ConvBlock(conv1_in_channels, conv1_out_channels, order, num_groups)
        self.conv2 = ConvBlock(conv2_in_channels, conv2_out_channels, order, num_groups)

        if conv1_in_channels != conv2_out_channels:
            self.skip_connection = fvnn.SparseConv3d(conv1_in_channels, conv2_out_channels, 1, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def _forward(self, input, hash_tree = None, feat_depth: int = 0):
        if self.use_pooling:
            if hash_tree is not None:
                feat_depth += 1
                input = self.maxpooling(input, hash_tree[feat_depth])
            else:
                input = self.maxpooling(input)
        
        h = input
        h = self.conv1(h)
        h = self.conv2(h)
        input = self.skip_connection(input)

        return h + input, feat_depth
    
    def forward(self, input, hash_tree = None, feat_depth: int = 0):
        if self.use_checkpoint:
            # !: we need to set use_reentrant = False
            input, feat_depth = checkpoint.checkpoint(self._forward, input, hash_tree, feat_depth, use_reentrant=False) 
        else:
            input, feat_depth = self._forward(input, hash_tree, feat_depth)

        if self.return_feat_depth:
            return input, feat_depth
        else:
            return input

class SparseDoubleConv(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 order: str,
                 num_groups: int,
                 encoder: bool,
                 pooling = None,
                 use_checkpoint: bool = False
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
            if pooling == 'max':
                self.add_module('MaxPool', fvnn.MaxPool(2))
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.add_module('SingleConv1', ConvBlock(conv1_in_channels, conv1_out_channels, order, num_groups))
        self.add_module('SingleConv2', ConvBlock(conv2_in_channels, conv2_out_channels, order, num_groups))
    
    def _forward(self, input, hash_tree = None, feat_depth: int = 0):
        for module in self:
            if module._get_name() == 'MaxPool' and hash_tree is not None:
                feat_depth += 1
                input = module(input, hash_tree[feat_depth])
            else:
                input = module(input)
        return input, feat_depth
    
    def forward(self, input, hash_tree = None, feat_depth: int = 0):
        if self.use_checkpoint:
            # !: we need to set use_reentrant = False
            input, feat_depth = checkpoint.checkpoint(self._forward, input, hash_tree, feat_depth, use_reentrant=False) 
        else:
            input, feat_depth = self._forward(input, hash_tree, feat_depth)
        return input, feat_depth

class AttentionBlock(nn.Module):
    """
    A for loop version with flash attention
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = fvnn.GroupNorm(32, channels)
        self.qkv = fvnn.Linear(channels, channels * 3)
        self.proj_out = fvnn.Linear(channels, channels)
        
    def _attention(self, qkv: torch.Tensor):
        # conduct attention for each batch
        length, width = qkv.shape
        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)
        qkv = qkv.reshape(length, self.num_heads, 3 * ch).unsqueeze(0)
        qkv = qkv.permute(0, 2, 1, 3) # (1, num_heads, length, 3 * ch)
        q, k, v = qkv.chunk(3, dim=-1) # (1, num_heads, length, ch)
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            values = F.scaled_dot_product_attention(q, k, v)[0] # (1, num_heads, length, ch)
        values = values.permute(1, 0, 2) # (length, num_heads, ch)
        values = values.reshape(length, -1)
        return values
        
    def attention(self, qkv: VDBTensor):
        values = []
        for batch_idx in range(qkv.grid.grid_count):
            values.append(self._attention(qkv.data[batch_idx].jdata))           
        return values  ### fvdb.JaggedTensor(values) ###

    def forward(self, x: VDBTensor):
        return self._forward(x), None # !: return None for feat_depth

    def _forward(self, x: VDBTensor):
        qkv = self.qkv(self.norm(x))
        values = self.attention(qkv)
        if qkv.grid.grid_count > 1:
            feature = fvdb.JaggedTensor(values)
        else:
            feature = x.grid.jagged_like(torch.cat(values, dim=0))   ##### buggy when batch size = 1, modified by myself, not sure ####
        feature = VDBTensor(x.grid, feature, x.kmap)
        feature = self.proj_out(feature)
        return feature + x
    

class Modified3DUnet(nn.Module):
    def __init__(self, lifter_params, 
                 in_channels, num_blocks, f_maps=64, order='gcs', num_groups=8,
                 neck_dense_type="UNCHANGED", neck_bound=4, 
                 with_render_branch=True,
                 gsplat_upsample=1, gs_enhanced="None",
                 use_attention=False, use_residual=True,
                 apply_gs_init: bool = True,
                 addtional_gs_constraint="None", 
                 use_checkpoint=False,
                 gs_init_scale=0.5,
                 gs_dim=14,
                 f_maps_2d=32,
                 feature_pooling_2d='max',
                 gs_free_space='hard',
                 max_return=1,
                 drop_invisible=False,
                 occ_upsample=2,
                 max_scaling=0.0,
                 **kwargs):
        super().__init__()
        self.lifter = Lifter(**lifter_params)

        n_features = [in_channels] + [f_maps * 2 ** k for k in range(num_blocks)]
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        self.pre_kl_bottleneck = nn.ModuleList()
        self.post_kl_bottleneck = nn.ModuleList()

        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.struct_convs = nn.ModuleList()
        self.num_blocks = num_blocks

        if not use_residual:
            basic_block = SparseDoubleConv
        else:
            basic_block = SparseResBlock

        # Attention setup
        self.use_attention = use_attention

        # Encoder
        self.pre_conv = fvnn.SparseConv3d(in_channels, in_channels, 1, 1) # a MLP to smooth the input
        for layer_idx in range(num_blocks):
            self.encoders.add_module(f'Enc{layer_idx}', basic_block(
                n_features[layer_idx], 
                n_features[layer_idx + 1], 
                order, 
                num_groups,
                True, # if encoder branch
                None,
                use_checkpoint
            ))
        for layer_idx in range(1, num_blocks):
            self.downsamplers.add_module(f'Down{layer_idx}', fvnn.MaxPool(2))

        # Bottleneck
        self.pre_kl_bottleneck.add_module(f'pre_kl_bottleneck_0', basic_block(
            n_features[-1], n_features[-1], order, num_groups, False, use_checkpoint=use_checkpoint))  
        if use_attention:
            self.pre_kl_bottleneck.add_module(f'pre_kl_attention', AttentionBlock(
                n_features[-1], use_checkpoint=use_checkpoint))
            # ! bug here -> forget to move this out
            self.pre_kl_bottleneck.add_module(f'pre_kl_bottleneck_1', basic_block(
                n_features[-1], n_features[-1], order, num_groups, False, use_checkpoint=use_checkpoint))

        self.post_kl_bottleneck.add_module(f'post_kl_bottleneck_0', basic_block(
            n_features[-1], n_features[-1], order, num_groups, False, use_checkpoint=use_checkpoint))
        if use_attention:
            self.post_kl_bottleneck.add_module(f'post_kl_attention', AttentionBlock(
                n_features[-1], use_checkpoint=use_checkpoint))
        self.post_kl_bottleneck.add_module(f'post_kl_bottleneck_1', basic_block(
            n_features[-1], n_features[-1], order, num_groups, False, use_checkpoint=use_checkpoint))
    
        # Decoder
        for layer_idx in range(-1, -num_blocks - 1, -1):
            self.struct_convs.add_module(f'Struct{layer_idx}', SparseHead(
                n_features[layer_idx], 2, order, num_groups))
            if layer_idx < -1:
                self.decoders.add_module(f'Dec{layer_idx}', basic_block(
                    n_features[layer_idx + 1] + n_features[layer_idx], # ! add back skip connection
                    n_features[layer_idx],
                    order, num_groups, False, None,
                    use_checkpoint=use_checkpoint
                ))
                self.upsamplers.add_module(f'Up{layer_idx}', fvnn.UpsamplingNearest(2))
        self.up_sample0 = fvnn.UpsamplingNearest(1)

        # check the type of neck_bound
        if isinstance(neck_bound, int):
            self.low_bound = [-neck_bound] * 3
            self.voxel_bound = [neck_bound * 2] * 3
        else:        
            self.low_bound = [-res for res in neck_bound]
            self.voxel_bound = [res * 2 for res in neck_bound]
        # self.neck_bound = neck_bound
        self.neck_dense_type = neck_dense_type
        
        self.with_render_branch = with_render_branch
        if with_render_branch:
            # ! hybrid head
            self.render_head_hybrid = LinearHead(n_features[1] + f_maps_2d, gsplat_upsample * gs_dim, order, num_groups, enhanced=gs_enhanced)
            if apply_gs_init:
                self.render_head_hybrid.OutConv.weight.data.zero_()
                init_value = self.render_head_hybrid.OutConv.bias.data.view(gsplat_upsample, gs_dim)
                init_value[:, :3] = 0.0
                if gsplat_upsample > 1:
                    init_value[:, :3] = torch.randn_like(init_value[:, :3]) * 0.5
                init_value[:, 3:6] = math.log(gs_init_scale)
                init_value[:, 6] = 1.0
                init_value[:, 7:10] = 0.0
                init_value[:, 10] = math.log(0.1 / (1 - 0.1))
                if gs_dim == 14: # rgb
                    init_value[:, 11:14] = 0.5
                self.render_head_hybrid.OutConv.bias.data = init_value.view(-1)

            # # ! 3D only head
            # self.render_head_3D = LinearHead(n_features[1], gsplat_upsample * gs_dim, order, num_groups, enhanced=gs_enhanced)
            # if apply_gs_init:
            #     self.render_head_3D.OutConv.weight.data.zero_()
            #     init_value = self.render_head_3D.OutConv.bias.data.view(gsplat_upsample, gs_dim)
            #     init_value[:, :3] = 0.0
            #     if gsplat_upsample > 1:
            #         init_value[:, :3] = torch.randn_like(init_value[:, :3]) * 0.5
            #     init_value[:, 3:6] = math.log(gs_init_scale)
            #     init_value[:, 6] = 1.0
            #     init_value[:, 7:10] = 0.0
            #     init_value[:, 10] = math.log(0.1 / (1 - 0.1))
            #     if gs_dim == 14: # rgb
            #         init_value[:, 11:14] = 0.5
            #     self.render_head_3D.OutConv.bias.data = init_value.view(-1)

        self.gsplat_upsample = gsplat_upsample
        self.addtional_gs_constraint = addtional_gs_constraint
        self.gs_dim = gs_dim

        self.padding = fvnn.FillFromGrid() # FillToGrid -> FillFromGrid
        # ! prepare for upsapmle occ-only part
        self.occ_upsample = fvnn.UpsamplingNearest(occ_upsample)
        self.feature_pooling_2d = feature_pooling_2d
        self.gs_free_space = gs_free_space
        self.max_return = max_return
        self.drop_invisible = drop_invisible
        self.max_scaling = max_scaling
        
    @classmethod
    def sparse_zero_padding(cls, in_x: fvnn.VDBTensor, target_grid: fvdb.GridBatch):
        source_grid = in_x.grid
        source_feature = in_x.data.jdata
        assert torch.allclose(source_grid.origins, target_grid.origins)
        assert torch.allclose(source_grid.voxel_sizes, target_grid.voxel_sizes)
        out_feat = torch.zeros((target_grid.total_voxels, source_feature.size(1)),
                               device=source_feature.device, dtype=source_feature.dtype)
        in_idx = source_grid.ijk_to_index(target_grid.ijk).jdata
        in_mask = in_idx != -1
        out_feat[in_mask] = source_feature[in_idx[in_mask]]
        return fvnn.VDBTensor(target_grid, target_grid.jagged_like(out_feat))
    
    @classmethod
    def struct_to_mask(cls, struct_pred: fvnn.VDBTensor):
        # 0 is exist, 1 is non-exist
        mask = struct_pred.data.jdata[:, 0] > struct_pred.data.jdata[:, 1]
        return struct_pred.grid.jagged_like(mask)

    @classmethod
    def cat(cls, x: fvnn.VDBTensor, y: fvnn.VDBTensor):
        assert x.grid == y.grid
        return fvnn.VDBTensor(x.grid, x.grid.jagged_like(torch.cat([x.data.jdata, y.data.jdata], dim=1)))
    
    def build_normal_hash_tree(self, input_grid):
        hash_tree = {}
        
        input_xyz = input_grid.grid_to_world(input_grid.ijk.float())
        _origins = input_grid.origins[0]
        _voxel_size = input_grid.voxel_sizes[0]
        
        for depth in range(self.num_blocks):            
            voxel_size = [sv * 2 ** depth for sv in _voxel_size]
            origins = [_origins[idx] + 0.5 * _voxel_size[idx] * (2 ** depth - 1) for idx in range(3)]
            
            if depth == 0:
                hash_tree[depth] = input_grid
            else:
                hash_tree[depth] = fvdb.gridbatch_from_nearest_voxels_to_points(
                    input_xyz, voxel_sizes=voxel_size, origins=origins)
        return hash_tree

    def build_fit_neck(self, sparse_grid, neck_expand: int = 1):
        sparse_coords = sparse_grid.ijk
        n_padding = (neck_expand - 1) // 2
        all_coords = []
        for b in range(sparse_grid.grid_count):
            min_bound = torch.min(sparse_coords[b].jdata, dim=0).values.cpu().numpy() - n_padding
            max_bound = torch.max(sparse_coords[b].jdata, dim=0).values.cpu().numpy() + 1 + n_padding
            cx = torch.arange(min_bound[0], max_bound[0], dtype=torch.int32, device=sparse_coords.device)
            cy = torch.arange(min_bound[1], max_bound[1], dtype=torch.int32, device=sparse_coords.device)
            cz = torch.arange(min_bound[2], max_bound[2], dtype=torch.int32, device=sparse_coords.device)
            coords = torch.stack(torch.meshgrid(cx, cy, cz, indexing='ij'), dim=3).view(-1, 3)
            all_coords.append(coords)
        all_coords = fvdb.JaggedTensor(all_coords)
        neck_grid = fvdb.gridbatch_from_ijk(all_coords,
                                              voxel_sizes=sparse_grid.voxel_sizes[0],
                                              origins=sparse_grid.origins[0])
        return neck_grid

    class FeaturesSet:
        def __init__(self):
            self.encoder_features = {}
            self.structure_features = {}
            self.structure_grid = {}
            self.render_features = {}

    def camera_intrinsic_list_to_matrix(self, intrinsic_list, normalize_pixel=False):
        """
        Args:
            intrinsic_list: [..., 6]
                [fx, fy, cx, cy, w, h]
        """
        if isinstance(intrinsic_list, list):
            intrinsic_list = torch.stack(intrinsic_list)
        fx, fy, cx, cy, w, h = intrinsic_list.unbind(-1)
        intrinsic_matrix = torch.zeros(intrinsic_list.shape[:-1] + (3, 3), device=intrinsic_list.device)

        intrinsic_matrix[..., 0, 0] = fx
        intrinsic_matrix[..., 1, 1] = fy
        intrinsic_matrix[..., 0, 2] = cx
        intrinsic_matrix[..., 1, 2] = cy
        intrinsic_matrix[..., 2, 2] = 1

        if normalize_pixel:
            intrinsic_matrix[..., 0, :] /= w[..., None]
            intrinsic_matrix[..., 1, :] /= h[..., None]

        return intrinsic_matrix
    
    def project_points(self, xyzs, proj_matrix):
        """
        Args:
            xyzs: jagged tensor, lshape [grid_num1, grid_num2], eshape [3]
            proj_matrix: [B, num_views, 3, 4], already normalized with h and w

        Returns:
            reference_points_cam: Jagged tensor, lshape [grid_num1, grid_num2], eshape [num_views, 1, 2]
            per_image_visibility_mask: Jagged tensor, lshape [grid_num1, grid_num2], eshape [num_views]
        """
        B = len(xyzs)
        V = proj_matrix.shape[1]
        reference_points_cam = []
        per_image_visibility_mask = []
        
        for b, xyz_jagged in enumerate(xyzs):
            xyz = xyz_jagged.jdata
            # Project 3D points to image space
            pts3d = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1) # grid_num_i, 4
            world_to_image = proj_matrix[b] # num_views, 3, 4
            pts2d = torch.einsum('vij,gj->gvi', world_to_image, pts3d) # grid_num_i, num_views, 3, fall in range [0,1]
            depth = pts2d[..., 2:]
            depth_valid_mask = depth > 0 # grid_num_i, num_views, 1
            uvs = pts2d[..., :2] / depth  # [0, 1]
            uv_valid_mask = (uvs >= 0) & (uvs <= 1) # grid_num_i, num_views, 2
            mask = depth_valid_mask[..., 0] & uv_valid_mask[..., 0] & uv_valid_mask[..., 1] # grid_num_i, num_views

            per_image_visibility_mask.append(mask)
            reference_points_cam.append(uvs.unsqueeze(2)) # grid_num_i, num_views, 1, 2

        reference_points_cam = JaggedTensor(reference_points_cam)
        per_image_visibility_mask = JaggedTensor(per_image_visibility_mask)

        return reference_points_cam, per_image_visibility_mask

    def get_rel_pos(self, _rel_xyz, free_space, grid):
        """
        Args:
            _rel_xyz: torch.tensor
                shape [..., 3]
            free_space: str
                something like 'soft'
            grid: GridBatch
                fvdb grid

        Returns:
            rel_pos: torch.tensor
                shape [..., 3], the position within a voxel (compared with the voxel center)
        """
        if free_space == "hard":
            rel_pos = torch.sigmoid(_rel_xyz) * grid.voxel_sizes[0]
        elif free_space == "soft":
            # free space [-1, 2]
            rel_pos = (torch.sigmoid(_rel_xyz) * 3 - 1) * grid.voxel_sizes[0]
        elif free_space == "soft-2":
            # free space [-2, 3]
            rel_pos = (torch.sigmoid(_rel_xyz) * 5 - 2) * grid.voxel_sizes[0]
        elif free_space == "soft-3":
            # free space [-3, 4]
            rel_pos = (torch.sigmoid(_rel_xyz) * 7 - 3) * grid.voxel_sizes[0]
        elif free_space == "soft-4":
            # free space [-4, 5]
            rel_pos = (torch.sigmoid(_rel_xyz) * 9 - 4) * grid.voxel_sizes[0]
        elif free_space == "soft-5":
            # free space [-5, 6]
            rel_pos = (torch.sigmoid(_rel_xyz) * 11 - 5) * grid.voxel_sizes[0]
        elif free_space == "tanh-3":
            # free space [-2.5, 3.5]
            rel_pos = (torch.tanh(_rel_xyz) * 3 + 0.5) * grid.voxel_sizes[0]
        elif free_space == "free-1":
            rel_pos = _rel_xyz * grid.voxel_sizes[0]
        elif free_space == "free-2":
            rel_pos = _rel_xyz
        elif free_space == "center":
            rel_pos = (torch.zeros_like(_rel_xyz) + 0.5) * grid.voxel_sizes[0]
        else:
            raise NotImplementedError 
        
        return rel_pos
    
    def camera_intrinsic_list_to_matrix(self, intrinsic_list, normalize_pixel=False):
        """
        Args:
            intrinsic_list: [..., 6]
                [fx, fy, cx, cy, w, h]
        """
        if isinstance(intrinsic_list, list):
            intrinsic_list = torch.stack(intrinsic_list)
        fx, fy, cx, cy, w, h = intrinsic_list.unbind(-1)
        intrinsic_matrix = torch.zeros(intrinsic_list.shape[:-1] + (3, 3), device=intrinsic_list.device)

        intrinsic_matrix[..., 0, 0] = fx
        intrinsic_matrix[..., 1, 1] = fy
        intrinsic_matrix[..., 0, 2] = cx
        intrinsic_matrix[..., 1, 2] = cy
        intrinsic_matrix[..., 2, 2] = 1

        if normalize_pixel:
            intrinsic_matrix[..., 0, :] /= w[..., None]
            intrinsic_matrix[..., 1, :] /= h[..., None]

        return intrinsic_matrix

    def _encode(self, x: fvnn.VDBTensor, hash_tree: dict, is_forward: bool = True):
        feat_depth = 0
        res = self.FeaturesSet()
        x = self.pre_conv(x)

        encoder_features = {}
        for module, downsampler in zip(self.encoders, [None] + list(self.downsamplers)):
            if downsampler is not None:
                x = downsampler(x, ref_coarse_data=hash_tree[feat_depth + 1])
                feat_depth += 1
            x, _ = module(x)
            encoder_features[feat_depth] = x

        if self.neck_dense_type == "UNCHANGED":
            pass
        elif self.neck_dense_type == "HAND_CRAFTED":
            voxel_size = x.grid.voxel_sizes[0] # !: modify for remain h
            origins = x.grid.origins[0] # !: modify for remain h
            neck_grid = fvdb.gridbatch_from_dense(
                x.grid.grid_count, 
                self.voxel_bound, 
                self.low_bound, # type: ignore
                device="cpu",
                voxel_sizes=voxel_size,
                origins=origins).to(x.device)
            x = fvnn.VDBTensor(neck_grid, neck_grid.fill_from_grid(x.data, x.grid, 0.0))
        elif self.neck_dense_type == "FIT":
            neck_grid = self.build_fit_neck(x.grid)
            x = self.padding(x, neck_grid)
        else:
            raise NotImplementedError

        for module in self.pre_kl_bottleneck:
            x, _ = module(x)
        return res, x, encoder_features
    
    def encode(self, x: fvnn.VDBTensor, hash_tree: dict):
        return self._encode(x, hash_tree, True)
    
    def decode(self, res: FeaturesSet, x: fvnn.VDBTensor, hash_tree: dict, 
               encoder_features: dict, img_features_batch, camera_pose, intrinsics):
        for module in self.post_kl_bottleneck:
            x, _ = module(x)
        
        struct_decision = None
        feat_depth = self.num_blocks - 1
        for module, upsampler, struct_conv in zip(
                [None] + list(self.decoders), [None] + list(self.upsamplers), self.struct_convs):  
            if module is not None:
                x = upsampler(x, struct_decision)
                feat_depth -= 1

                enc_feat = self.padding(encoder_features[feat_depth], x)
                x = fvdb.jcat([enc_feat, x], dim=1)
                x, _ = module(x)
            # guided setting do not need to predict structure
            res.structure_features[feat_depth] = None
            # get the guided structure
            target_struct = hash_tree[feat_depth]
            struct_decision = target_struct.ijk_to_index(x.grid.ijk).jdata > -1
            res.structure_grid[feat_depth] = self.up_sample0(x, struct_decision).grid

        x = self.up_sample0(x, struct_decision)

        position_list = []
        opacity_list = []
        scaling_list = []
        rotation_list = []
        feat_dc_list = []
        if self.with_render_branch:
            if x.grid.total_voxels > 0:
                h, w = img_features_batch.shape[3:5]
                grid = x.grid

                world_to_camera = torch.inverse(camera_pose) # [B, N, 4, 4]
                input_camera_K = self.camera_intrinsic_list_to_matrix(intrinsics, normalize_pixel=True) # [B, N, 3, 3]
                world_to_image = torch.einsum('bnij,bnjk->bnik', input_camera_K, world_to_camera[...,:3,:4]) # [B, N, 3, 4]
     
                for bidx in range(grid.grid_count):
                    cur_grid = grid[bidx]
                    image_features = img_features_batch[bidx] # N, C, H, W
                    torch.cuda.empty_cache()

                    # ! process occ-only part
                    ## use voxel center xyz to query image feature and then concat with voxel feature
                    cur_occ_tensor = self.occ_upsample(VDBTensor(cur_grid, x.data[bidx]))

                    cur_occ_grid = cur_occ_tensor.grid
                    cur_occ_xyz = cur_occ_grid.grid_to_world(cur_occ_grid.ijk.float())

                    # camera coordinate
                    reference_points_cam, per_image_visibility_mask = \
                        self.project_points(cur_occ_xyz, world_to_image[bidx:bidx+1])

                    # !! since we are in batch = 1
                    reference_points_cam = reference_points_cam.jdata # [N_voxel, N_view, 1, 2]
                    reference_points_cam = reference_points_cam.permute(1,0,2,3) # [N_view, N_voxel, 1, 2]. pseduo height = N_voxel, width = 1
                    grid_to_sample = 2 * reference_points_cam - 1 # [N_view, N_voxel, 1, 2]

                    if self.feature_pooling_2d == 'max':
                        sampled_features = F.grid_sample(image_features, grid_to_sample) # [N_view, C, N_voxel, 1]
                        sampled_features = sampled_features[..., 0].permute(2, 0, 1) # [N_voxel, N_view, C]

                        # mask out invisible points in some camera, using occ_front_per_camera
                        occ_voxel_2D_feature = torch.max(sampled_features, dim=1)[0]
                    else:
                        raise NotImplementedError
                    
                    occ_voxel_3D_feature = cur_occ_tensor.data.jdata

                    occ_voxel_hybrid_feature = torch.cat([occ_voxel_2D_feature, occ_voxel_3D_feature], dim=1)
                    occ_render_feature = self.render_head_hybrid(VDBTensor(cur_occ_grid, cur_occ_grid.jagged_like(occ_voxel_hybrid_feature)))
                    occ_abs_pos, occ_scaling, occ_rotation, occ_opacity, occ_color = self.feature2gs(cur_occ_grid, occ_render_feature.data.jdata, gs_free_space=self.gs_free_space, max_scaling=self.max_scaling)

                    # non_occ_render_feature = self.render_head_3D(VDBTensor(cur_occ_grid, cur_occ_grid.jagged_like(occ_voxel_3D_feature)))
                    # abs_pos, scaling, rotation, opacity, color = self.feature2gs(cur_occ_grid, non_occ_render_feature.data.jdata, gs_free_space=self.gs_free_space, max_scaling=self.max_scaling)

                    # position_list.append(torch.cat([abs_pos, occ_abs_pos], dim=0))
                    # scaling_list.append(torch.cat([scaling, occ_scaling], dim=0))
                    # rotation_list.append(torch.cat([rotation, occ_rotation], dim=0))
                    # opacity_list.append(torch.cat([opacity, occ_opacity], dim=0))
                    # feat_dc_list.append(torch.cat([color, occ_color], dim=0))  

                    # position_list.append(abs_pos)
                    # scaling_list.append(scaling)
                    # rotation_list.append(rotation)
                    # opacity_list.append(opacity)
                    # feat_dc_list.append(color) 

                    position_list.append(occ_abs_pos)
                    scaling_list.append(occ_scaling)
                    rotation_list.append(occ_rotation)
                    opacity_list.append(occ_opacity)
                    feat_dc_list.append(occ_color)    

        return position_list, opacity_list, scaling_list, rotation_list, feat_dc_list
    
    def feature2gs(self, grid, feature, gs_free_space= "hard", max_scaling = 1):
        """split the gaussian parameters (default)
        xyz: 3 channel
        scale: 3 channel
        rotation: 4 channel
        opacity: 1 channel
        color: 3 channel
        """
        feature = feature.view(-1, self.gsplat_upsample, self.gs_dim)
        _rel_xyz = feature[:, :, :3]
        _scaling = feature[:, :, 3:6]
        _rots = feature[:, :, 6:10]
        _opacities = feature[:, :, 10:11]
        _color = feature[:, :, 11:self.gs_dim]

        rel_pos = self.get_rel_pos(_rel_xyz, gs_free_space, grid)
        abs_pos = grid.grid_to_world(grid.ijk.float() - 0.5).jdata.unsqueeze(1) + rel_pos
        abs_pos = abs_pos.view(-1, 3)
        scaling = (torch.exp(_scaling) * grid.voxel_sizes[0, 0]).view(-1, 3)
        
        if max_scaling > 0.0:
            scaling = torch.clamp(scaling, max=max_scaling)
            
        rotation = torch.nn.functional.normalize(_rots.view(-1, 4), dim=1)
        opacity = torch.sigmoid(_opacities.view(-1, 1))

        color_dim = self.gs_dim - 11
        color = _color.view(-1, color_dim) # rgb or feature

        return abs_pos.float(), scaling, rotation, opacity, color.unsqueeze(1) # for rasterizer
    
    def forward(self, coors, padding_coors, voxel_sizes, origins, voxel_features, img_features, camera_pose, intrinsics, depths, padding_confidence):
        # coors list[N, 3]
        # padding coors list[N2, 3]
        # voxel_sizes [x, x, x]
        # origins [x, x, x]
        # voxel_features list[N, C]        
        # img_features [B, NUM_VIEW, C, H, W]        
        # camera_pose [B, NUM_VIEW, 4, 4]
        # intrinsics  [B, NUM_VIEW, 3, 3]
        # depths [B NUM_VIEW C H W]
        # confidnence [N 1]
        
        batch_size = intrinsics.shape[0]

        padded_img_features = img_features.clone()
        padded_intrinsics = intrinsics.clone()

        if padding_coors:
            padding_size = 56
            B, N, C, H, W = padded_img_features.shape
            padded_img_features = padded_img_features.view(B * N, C, H, W)  # Flatten B and N into one batch
            padded_img_features = F.pad(padded_img_features, (padding_size, padding_size, padding_size, padding_size), mode="replicate")
            padded_img_features = padded_img_features.view(B, N, C, padded_img_features.shape[-2], padded_img_features.shape[-1])
            B, N, C, H, W = padded_img_features.shape
            padded_intrinsics[..., 2] += padding_size  # cx (center x)
            padded_intrinsics[..., 3] += padding_size  # cy (center y)
            padded_intrinsics[..., 4] = W 
            padded_intrinsics[..., 5] = H

            padding_grid = fvdb.gridbatch_from_ijk(
                fvdb.JaggedTensor(padding_coors),  
                voxel_sizes=[voxel_sizes for _ in range(batch_size)],
                origins=[origins for _ in range(batch_size)]
            )

            padding_voxel_features = self.lifter(padding_grid, camera_pose, padded_intrinsics, padded_img_features, depths, padding_confidence)
            del padding_grid

            for b in range(batch_size):
                coors[b] = torch.cat([coors[b], padding_coors[b]], dim=0)
                voxel_features[b] = torch.cat([voxel_features[b], padding_voxel_features[b]], dim=0)

        x_grid = fvdb.gridbatch_from_ijk(
            fvdb.JaggedTensor(coors),  
            voxel_sizes=[voxel_sizes for _ in range(batch_size)],
            origins=[origins for _ in range(batch_size)]
        )
        allbatch_voxel_features = torch.cat(voxel_features, dim=0)

        x = fvnn.VDBTensor(x_grid, x_grid.jagged_like(allbatch_voxel_features))

        n_imgs, H, W = padded_img_features.size(1), padded_img_features.size(3), padded_img_features.size(4)
        if (H != padded_intrinsics[..., 5]).all() or (W != padded_intrinsics[..., 4]).all():
            print("intrinsic HWs has problem!")
            downsample_h = padded_intrinsics[0, 0, 5] / H
            downsample_w = padded_intrinsics[0, 0, 4] / W
            padded_intrinsics[..., [1,3,5]] = padded_intrinsics[..., [1,3,5]] / downsample_h
            padded_intrinsics[..., [0,2,4]] = padded_intrinsics[..., [0,2,4]] / downsample_w

        # build a hash tree
        hash_tree = self.build_normal_hash_tree(x.grid)
        res, x, encoder_features = self.encode(x, hash_tree)
        position_list, opacity_list, scaling_list, rotation_list, feat_dc_list = self.decode(res, x, hash_tree, encoder_features, padded_img_features, camera_pose, padded_intrinsics)
        
        del padded_img_features
        del padded_intrinsics

        return position_list, opacity_list, scaling_list, rotation_list, feat_dc_list


class Lifter(nn.Module):
    def __init__(self, img_in_dim, voxel_out_dim):
        super().__init__()
        self.mix_fc = nn.Linear(img_in_dim, voxel_out_dim)

    def create_rays_from_intrinsic_torch_batch(self, pose_matric, intrinsic):
        """
        Args:
            pose_matric: (B, 4, 4)
            intrinsic: (B, 6), [fx, fy, cx, cy, w, h]
        Returns:
            camera_origin: (B, 3)
            d: (B, H, W, 3)
        """
        camera_origin = pose_matric[:, :3, 3] # (B, 3)
        fx, fy, cx, cy, w, h = intrinsic.unbind(1) # [B,]
        w, h = int(w[0]), int(h[0])
        # attention, indexing is 'xy'
        ii, jj = torch.meshgrid(torch.arange(w).to(intrinsic.device), torch.arange(h).to(intrinsic.device), indexing='xy') 

        ii = ii[None].repeat(pose_matric.shape[0], 1, 1) # (B, H, W)
        jj = jj[None].repeat(pose_matric.shape[0], 1, 1) # (B, H, W)

        uu, vv = (ii - cx[:, None, None]) / fx[:, None, None], (jj - cy[:, None, None]) / fy[:, None, None]
        local_xyz = torch.stack([uu, vv, torch.ones_like(uu, device=uu.device)], dim=-1) # (B, H, W, 3)
        local_xyz = torch.cat([local_xyz, torch.ones((local_xyz.shape[0], int(h), int(w), 1)).to(local_xyz)], axis=-1)
        pixel_xyz = torch.einsum('bij, bhwj->bhwi', pose_matric, local_xyz)[:, :, :, :3] # (B, H, W, 3) # ! fix error

        d = (pixel_xyz - camera_origin[:, None, None, :])  # (B, H, W, 3)
        # normalize the direction
        d = d / torch.norm(d, dim=-1, keepdim=True) # (B, H, W, 3)

        return camera_origin, d

    def build_ray_casting_feature(self, grid, camera_pose, padded_intrinsics, padded_img_features):
        """
        This is previous `build_occulusion_feature_cube`, I use the new name for more accurate meaning

        We unproject the image pixels to the voxel grid, and assign the pixel feature to the voxel grid,
        then return the voxel feature for each voxel.

        Args:
            grid: len(grid) == B
            img_features: [B, N, C, H, W]
            camera_pose: [B, N, 4, 4]
            intrinsics: [B, N, 6], 6 is fx fy cx cy w h

        Returns:
            voxel_features: JaggedTensor
        """

        voxel_features = []
        n_imgs, H, W = padded_img_features.size(1), padded_img_features.size(3), padded_img_features.size(4)
        # update the batch[DS.IMAGES_INPUT_INTRINSIC]
        if (H != padded_intrinsics[..., 5]).all() or (W != padded_intrinsics[..., 4]).all():
            print("intrinsic HW doesn't match!")
            downsample_h = padded_intrinsics[0, 0, 5] / H
            downsample_w = padded_intrinsics[0, 0, 4] / W
            padded_intrinsics[..., [1,3,5]] = padded_intrinsics[..., [1,3,5]] / downsample_h
            padded_intrinsics[..., [0,2,4]] = padded_intrinsics[..., [0,2,4]] / downsample_w

        for bidx in range(grid.grid_count):
            cur_grid = grid[bidx]
            cur_pose = camera_pose[bidx] # N, 4, 4
            cur_intrinsics = padded_intrinsics[bidx]

            # [N, 3], [N, H, W, 3] -> [N * H * W, 3]
            nimg_origins, nimg_directions = self.create_rays_from_intrinsic_torch_batch(cur_pose, cur_intrinsics)
            nimg_origins = nimg_origins.view(n_imgs, 1, 1, 3).expand(-1, H, W, -1).reshape(-1, 3)
            nimg_directions = nimg_directions.reshape(-1, 3)

            nimg_features = padded_img_features[bidx] # N, C, H, W
            nimg_features = nimg_features.permute(0, 2, 3, 1).view(n_imgs * H * W, -1) # N, C, H, W -> N, H, W, C -> N * H * W, C

            if fvdb.__version__ == '0.0.0':
                pack_info, out_voxel_ijk, _ = cur_grid.voxels_along_rays(JaggedTensor([nimg_origins]), 
                                                                         JaggedTensor([nimg_directions]), 
                                                                         max_voxels=1)
                out_voxel_ids = cur_grid.ijk_to_index(out_voxel_ijk)
                pixel_feature = nimg_features[pack_info.jdata[:, 1] > 0, :]
            else:
                out_voxel_ids, ray_start_end = cur_grid.voxels_along_rays(JaggedTensor([nimg_origins]), 
                                                                         JaggedTensor([nimg_directions]), 
                                                                         max_voxels=1, 
                                                                         return_ijk=False)

                mask = (ray_start_end.joffsets[1:] - ray_start_end.joffsets[:-1]).bool() # [N_ray]
                pixel_feature = nimg_features[mask, :] # [N_ray_hit, C]
                out_voxel_ids = out_voxel_ids.jdata.to(torch.int64)
                
            out_voxel_features = torch.zeros((cur_grid.total_voxels, nimg_features.shape[1]), device=cur_pose.device)
            out_voxel_features = scatter_mean(pixel_feature, out_voxel_ids, out=out_voxel_features, dim=0)

            voxel_features.append(out_voxel_features)

        return voxel_features

    def forward(self, grid, camera_pose, padded_intrinsics, padded_img_features, depths, padding_confidence):
        # img_features = torch.cat([img_features, depths], dim=2)
        voxel_features = self.build_ray_casting_feature(grid, camera_pose, padded_intrinsics, padded_img_features)
        for b in range(len(voxel_features)):
            voxel_features[b] = torch.cat([voxel_features[b], padding_confidence[b]], dim=-1)
        voxel_features = [self.mix_fc(feature) for feature in voxel_features]
        return voxel_features