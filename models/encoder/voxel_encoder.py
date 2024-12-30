import torch
from torch_scatter import scatter_min, scatter_add
from torch.nn import functional as F
from torch import Tensor, nn

import torch

def compute_voxel_coors_and_centers(
    pts3d: torch.Tensor,      # (N, 3)
    voxel_size: float = 0.02,
    point_cloud_range=(0, 0, 0, 0, 0, 0)
):
    device = pts3d.device
    
    # 1) 计算体素坐标
    #    voxel_indices[i] = floor((pts3d[i] - grid_min) / voxel_size)
    grid_min = torch.tensor(point_cloud_range[:3], dtype=torch.float32, device=device)
    grid_max = torch.tensor(point_cloud_range[3:], dtype=torch.float32, device=device)
    voxel_indices = torch.floor((pts3d - grid_min) / voxel_size).long()  # (N, 3)
    
    # 2) 筛选出范围内的点
    voxel_dims = ((grid_max - grid_min) / voxel_size).long()  # (3,)
    mask_min = (voxel_indices >= 0).all(dim=1)
    mask_max = (voxel_indices < voxel_dims).all(dim=1)
    valid_mask = mask_min & mask_max
    
    pts3d_valid = pts3d[valid_mask]
    voxel_indices_valid = voxel_indices[valid_mask]
    
    # 3) 为体素分配一个哈希 (z*1e6 + y*1e3 + x)
    voxel_hash = (
          voxel_indices_valid[:, 2] * 10**6
        + voxel_indices_valid[:, 1] * 10**3
        + voxel_indices_valid[:, 0]
    )
    
    # 4) 找到去重后的体素哈希，以及对应下标
    unique_voxel_hash = torch.unique(voxel_hash)
    M = unique_voxel_hash.size(0)  # 体素数
    
    # 5) 反推得到 x, y, z 并构造 coors
    #    注意: z = hash // 1e6, 然后再取 yz = hash % 1e6, y = yz // 1e3, x = yz % 1e3
    z = (unique_voxel_hash // 10**6).long()
    yz = (unique_voxel_hash % 10**6).long()
    y = (yz // 10**3).long()
    x = (yz % 10**3).long()
    
    # batch_idx 这里全用 0
    batch_idx = torch.zeros_like(z, dtype=torch.long)
    
    # (M, 4): [batch_idx, z, y, x]
    coors = torch.stack([batch_idx, z, y, x], dim=1)  # (M,4)
    
    # 6) 计算体素中心
    #    center_x = x * voxel_size + grid_min[0] + voxel_size/2
    #    center_y = ...
    #    center_z = ...
    voxel_centers = torch.stack([
        x.float() * voxel_size + grid_min[0] + voxel_size / 2,
        y.float() * voxel_size + grid_min[1] + voxel_size / 2,
        z.float() * voxel_size + grid_min[2] + voxel_size / 2
    ], dim=1)  # (M, 3)
    
    return coors, voxel_centers



def prepare_hard_vfe_inputs_scatter_fast(
    pts3d: torch.Tensor,      # (N, 3)
    pts_feat: torch.Tensor,   # (N, D)    
    pts_rgb: torch.Tensor,    # (N, 3)
    voxel_size: float = 0.02,
    point_cloud_range=(0, 0, 0, 0, 0, 0),
    max_points: int = 5
):
    """
    将点云坐标、特征、颜色打包到 [M, max_points, C] 的张量里
    并返回:
       features:   (M, max_points, C)  其中 C = 3(xyz) + D(特征) + 3(rgb)
       num_points: (M, )              每个voxel中的实际点数 (未截断)
       coors:      (M, 4)            [batch_idx, z, y, x] 的体素坐标
    """
    device = pts3d.device
    N = pts3d.size(0)
    # 1) 计算体素坐标
    grid_min = torch.tensor(point_cloud_range[:3], dtype=torch.float32, device=device)
    grid_max = torch.tensor(point_cloud_range[3:], dtype=torch.float32, device=device)
    voxel_indices = torch.floor((pts3d - grid_min) / voxel_size).long()  # (N, 3)

    # 2) 筛选出范围内的点
    voxel_dims = ((grid_max - grid_min) / voxel_size).long()  # x,y,z 每个最多的体素数量
    mask_min = (voxel_indices >= 0).all(dim=1)
    mask_max = (voxel_indices < voxel_dims).all(dim=1)
    valid_mask = mask_min & mask_max

    pts3d       = pts3d[valid_mask]
    pts_feat    = pts_feat[valid_mask]
    pts_rgb     = pts_rgb[valid_mask]
    voxel_indices = voxel_indices[valid_mask]

    # 3) 为体素分配一个哈希
    #    这里约定: voxel_hash = z * 1e6 + y * 1e3 + x
    #    注意 x=voxel_indices[:,0], y=1, z=2 (与 HardVFE 的 coors[:,1]~3 对应)
    voxel_hash = (
          voxel_indices[:, 2] * 10**6  # z
        + voxel_indices[:, 1] * 10**3  # y
        + voxel_indices[:, 0]         # x
    )

    # 4) 找到去重后的哈希，以及每个点对应的 voxel_id
    unique_voxel_hash, inverse_indices = torch.unique(voxel_hash, return_inverse=True)
    M = unique_voxel_hash.size(0)  # 体素数

    # 拼接点特征: [x, y, z, feat..., rgb...]
    scatter_features = torch.cat([pts3d, pts_feat, pts_rgb], dim=1)  # (N_valid, C)
    C = scatter_features.size(1)

    # --------------- 核心加速步骤 --------------
    # Step A: 按照 voxel_id (即 inverse_indices) 排序，分组放在一起
    sorted_indices = torch.argsort(inverse_indices)            # (N_valid, )
    sorted_voxel_ids = inverse_indices[sorted_indices]         # 体素ID 已按升序排列
    sorted_features = scatter_features[sorted_indices]         # (N_valid, C)

    # Step B: 计算每个点在本体素内的“偏移量 offset”
    #   1) 先给每组打 group_id (0 ~ M-1)
    #      diff[i] 表示 sorted_voxel_ids[i] 与前一个是否不一样
    #      然后 cumsum 就得到一个组号 group_id[i]。
    diff = (sorted_voxel_ids[1:] != sorted_voxel_ids[:-1])
    group_id = torch.zeros_like(sorted_voxel_ids)
    if group_id.numel() > 1:
        group_id[1:] = diff.cumsum(dim=0)
    #   2) 计算每个 group 的第一个全局索引 => group_min[g] = 该组在 sorted_voxel_ids 里出现的第一个下标
    idx_arange = torch.arange(group_id.size(0), device=device)
    group_min, _ = scatter_min(idx_arange, group_id, dim=0, dim_size=M)
    #   3) offset = 当前下标 - 本组起始下标 => 即组内位置
    offsets = idx_arange - group_min[group_id]
    #   4) clamp offset 到 max_points-1
    offsets_clamped = offsets.clamp_max(max_points - 1)

    # Step C: 一次性写入 [M, max_points, C]
    features = torch.zeros((M, max_points, C), dtype=torch.float32, device=device)
    features[group_id.long(), offsets_clamped.long()] = sorted_features

    # Step D: 统计每个 voxel 的实际点数 (未截断)
    #         如果只想要“截断后”的点数，可以再 clamp 一下，但 HardVFE 通常需要真实数量
    ones = torch.ones_like(group_id, dtype=torch.int32)
    num_points = scatter_add(ones, group_id, dim=0, dim_size=M)
    # num_points_clamped = num_points.clamp(max=max_points) # 若你只想要截断后的计数，可以再做这一步

    # 5) 反推 x,y,z，用于构造 coors (batch=0)
    z = (unique_voxel_hash // 10**6).long()
    yz = (unique_voxel_hash % 10**6).long()
    y = (yz // 10**3).long()
    x = (yz % 10**3).long()

    batch_idx = torch.zeros_like(z, dtype=torch.long)
    # HardVFE 的惯例: coors[:,0]=batch_idx, coors[:,1]=z, coors[:,2]=y, coors[:,3]=x
    coors = torch.stack([batch_idx, z, y, x], dim=1)  # (M,4)
    
    voxel_centers = torch.stack([
        x.float() * voxel_size + grid_min[0] + voxel_size / 2,
        y.float() * voxel_size + grid_min[1] + voxel_size / 2,
        z.float() * voxel_size + grid_min[2] + voxel_size / 2
    ], dim=1)  # [M, 3]

    return features, num_points, coors, voxel_centers

def get_paddings_indicator(actual_num: Tensor,
                           max_num: Tensor,
                           axis: int = 0) -> Tensor:
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class VFELayer(nn.Module):
    """Voxel Feature Encoder layer.

    The voxel encoder is composed of a series of these layers.
    This module do not support average pooling and only support to use
    max pooling to gather features inside a VFE.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
        max_out (bool): Whether aggregate the features of points inside
            each voxel and only return voxel features.
        cat_max (bool): Whether concatenate the aggregated features
            and pointwise features.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 max_out: bool = True,
                 cat_max: bool = True):
        super(VFELayer, self).__init__()
        self.cat_max = cat_max
        self.max_out = max_out
        # self.units = int(out_channels / 2)

        # self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.norm =  nn.BatchNorm1d(out_channels)
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward function.

        Args:
            inputs (torch.Tensor): Voxels features of shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.

        Returns:
            torch.Tensor: Voxel features. There are three mode under which the
                features have different meaning.
                - `max_out=False`: Return point-wise features in
                    shape (N, M, C).
                - `max_out=True` and `cat_max=False`: Return aggregated
                    voxel features in shape (N, C)
                - `max_out=True` and `cat_max=True`: Return concatenated
                    point-wise features in shape (N, M, C).
        """
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        pointwise = F.relu(x)
        # [K, T, units]
        if self.max_out:
            aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        else:
            # this is for fusion layer
            return pointwise

        if not self.cat_max:
            return aggregated.squeeze(1)
        else:
            # [K, 1, units]
            repeated = aggregated.repeat(1, voxel_count, 1)
            concatenated = torch.cat([pointwise, repeated], dim=2)
            # [K, T, 2 * units]
            return concatenated

class HardVFE(nn.Module):
    """Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.

    Args:
        in_channels (int, optional): Input channels of VFE. Defaults to 4.
        feat_channels (list(int), optional): Channels of features in VFE.
        with_distance (bool, optional): Whether to use the L2 distance
            of points to the origin point. Defaults to False.
        with_cluster_center (bool, optional): Whether to use the distance
            to cluster center of points inside a voxel. Defaults to False.
        with_voxel_center (bool, optional): Whether to use the distance to
            center of voxel for each points inside a voxel. Defaults to False.
        voxel_size (tuple[float], optional): Size of a single voxel.
            Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): The range of points
            or voxels. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict, optional): Config dict of normalization layers.
        mode (str, optional): The mode when pooling features of points inside a
            voxel. Available options include 'max' and 'avg'.
            Defaults to 'max'.
        fusion_layer (dict, optional): The config dict of fusion layer
            used in multi-modal detectors. Defaults to None.
        return_point_feats (bool, optional): Whether to return the
            features of each points. Defaults to False.
    """

    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: list = [],
                 with_distance: bool = False,
                 with_cluster_center: bool = False,
                 with_voxel_center: bool = False,
                 voxel_size: tuple = (0.2, 0.2, 4),
                 point_cloud_range: tuple = (0, -40, -3, 70.4, 40, 1),
                 mode: str = 'max',
                 fusion_layer: dict = None,
                 return_point_feats: bool = False):
        super(HardVFE, self).__init__()
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats

        # Need pillar (voxel) size and x/y offset to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            # TODO: pass norm_cfg to VFE
            # norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            if i == (len(feat_channels) - 2):
                cat_max = False
                max_out = True
                if fusion_layer:
                    max_out = False
            else:
                max_out = True
                cat_max = True
            vfe_layers.append(
                VFELayer(
                    in_filters,
                    out_filters,
                    max_out=max_out,
                    cat_max=cat_max))
            self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)


    def forward(self,
                features: Tensor,
                num_points: Tensor,
                coors: Tensor,
                *args,
                **kwargs) -> tuple:
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is MxNxC.
            num_points (torch.Tensor): Number of points in each voxel.
            coors (torch.Tensor): Coordinates of voxels, shape is Mx(1+NDim).
            img_feats (list[torch.Tensor], optional): Image features used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = (
                features[:, :, :3].sum(dim=1, keepdim=True) /
                num_points.type_as(features).view(-1, 1, 1))
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(
                size=(features.size(0), features.size(1), 3))
            f_center[:, :, 0] = features[:, :, 0] - (
                coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                self.x_offset)
            f_center[:, :, 1] = features[:, :, 1] - (
                coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                self.y_offset)
            f_center[:, :, 2] = features[:, :, 2] - (
                coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        voxel_feats = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty.
        # Need to ensure that empty voxels remain set to zeros.
        voxel_count = voxel_feats.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        voxel_feats *= mask.unsqueeze(-1).type_as(voxel_feats)

        for i, vfe in enumerate(self.vfe_layers):
            voxel_feats = vfe(voxel_feats)

        return voxel_feats