"""
Point Transformer V2 Mode 2 (recommend)

Disable Grouped Linear

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid, fps
from torch_scatter import segment_csr

import einops
from timm.models.layers import DropPath
import pointops
from IPython import embed

import time

class TimedModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *inputs, **kwargs):
        start_time = time.time()  # 记录开始时间
        output = self.module(*inputs, **kwargs)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        print(f"Module: {self.module.__class__.__name__}, Time taken: {elapsed_time:.6f} seconds")
        return output

class Timer:
    def __init__(self, name="Timer"):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()  # 记录开始时间
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()  # 记录结束时间
        self.elapsed_time = self.end_time - self.start_time
        print(f"{self.name}: Time taken: {self.elapsed_time:.6f} seconds")
        pass

@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.no_grad()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class PointLayerNorm(nn.Module):
    """
    Layer Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.LayerNorm(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class MLP(nn.Module):
    '''
    A standard MLP.
    '''
    def __init__(self, in_channels, out_channels, layers=2, hidden_channels=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.layers = nn.ModuleList([nn.Linear(in_channels, hidden_channels)])
        self.layers.append(act_layer())
        self.layers.append(nn.Dropout(drop))
        for i in range(layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.layers.append(act_layer())
            self.layers.append(nn.Dropout(drop))
    
        self.layers.append(nn.Linear(hidden_channels, out_channels))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionDownFPS(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, nsample=16, norm_layer=nn.LayerNorm):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_channels, out_channels, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            
            assert p.dtype == torch.float32, "Point coordinates should be float32"
            idx = pointops.farthest_point_sampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            
            with torch.no_grad():
                assert p.dtype == torch.float32, "Point coordinates should be float32"
                idx, _ = pointops.knn_query(
                    self.nsample, p, o, n_p, n_o
                )
            
            x = pointops.grouping(idx, x, p, n_p, with_xyz=True)
            x = self.relu(self.norm(self.linear(x))) # (m, nsample, c)
            x = einops.rearrange(x, "m n c -> m c n")
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.norm(self.linear(x)))  # (n, c)
        return [p, x, o], None


class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = (
            segment_csr(
                coord,
                torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                reduce="min",
            )
            if start is None
            else start
        )
        cluster = voxel_grid(
            pos=coord - start[batch], size=self.grid_size, batch=batch, start=0
        )
        unique, cluster, counts = torch.unique(
            cluster, sorted=True, return_inverse=True, return_counts=True
        )
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset], cluster


class Expand(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, norm_layer=nn.LayerNorm, identity=True):
        super().__init__()
        self.expansion = expansion
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.identity = identity
        
        if not identity:
            self.expand_feat = MLP(3 + in_channels, out_channels * expansion)
            self.expand_offset = MLP(3 + in_channels, 3 * expansion)
        else:
            self.expand_feat = MLP(3 + in_channels, out_channels * expansion)
            self.expand_offset = MLP(3 + in_channels, 3 * (expansion - 1))
        
        self.act = nn.ReLU(inplace=True)
        self.norm = norm_layer(out_channels)
    
    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c_in), (b)
        offsets = self.expand_offset(torch.cat([p, x], dim=-1)) # (n, 3*expansion)
        if not self.identity:
            offsets = einops.rearrange(offsets, "n (r c) -> n r c", r=self.expansion) # (n, expansion, 3)
        else:
            offsets = einops.rearrange(offsets, "n (r c) -> n r c", r=self.expansion - 1) # (n, expansion-1, 3)
            offsets = torch.cat([torch.zeros_like(offsets[:, :1, :]), offsets], dim=1) # (n, expansion, 3)
            # offsets = torch.zeros_like(offsets[:, :, :])
        
        new_p = p.unsqueeze(1) + offsets # (n, expansion, 3)
        new_p = new_p.reshape(-1, 3) # (n*expansion, 3)

        new_feats = self.expand_feat(torch.cat([p, x], dim=-1)) # (n, c_out*expansion)
        new_feats = einops.rearrange(new_feats, "n (r c) -> (n r) c", r=self.expansion) # (n*expansion, c_out)
        new_feats = self.act(self.norm(new_feats)) # (n*expansion, c_out)

        new_o = o * self.expansion # (b), each sample's point number is multiplied by expansion

        return [new_p, new_feats, new_o]


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels=None, nsample=3, norm_layer=nn.LayerNorm):
        super().__init__()
        self.nsample = nsample
        if out_channels is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2 * in_channels, in_channels),
                norm_layer(in_channels),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_channels, in_channels), nn.ReLU(inplace=True)
            )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                norm_layer(out_channels),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                norm_layer(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, pxo1, pxo2=None):
        '''
            pxo1: (p1, x1, o1), query points
            pxo2: (p2, x2, o2), points to be interpolated
        '''
        if pxo2 is None:
            p, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat(
                    (x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1
                )
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
            return [p, x, o]
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(
                p2, p1, self.linear2(x2), o2, o1, self.nsample
            )
            return [p1, x, o1]


class RandomDownSample(nn.Module):
    def __init__(self, down_ratio=2):
        super().__init__()
        self.down_ratio = down_ratio
        assert down_ratio >= 1
    
    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        n_o = []
        n_p = []
        n_x = []
        cumsum = 0
        b = offset2bincount(o)
        for i in range(b.size(0)):
            count = b[i]
            count_left = count // self.down_ratio
            idx = torch.randperm(count)[:count_left]
            start = o[i - 1] if i > 0 else 0
            
            n_p.append(p[start:start+count][idx])
            n_x.append(x[start:start+count][idx])

            cumsum += count_left
            n_o.append(cumsum)

        n_o = torch.tensor(n_o, device=o.device, dtype=torch.int32)
        n_p = torch.cat(n_p, dim=0)
        n_x = torch.cat(n_x, dim=0)

        return [n_p, n_x, n_o]


class GroupedVectorAttention(nn.Module):
    def __init__(
        self,
        embed_channels,
        groups,
        attn_drop_rate=0.0,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        norm_layer=nn.LayerNorm,
    ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            norm_layer(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            norm_layer(embed_channels),
            nn.ReLU(inplace=True),
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                norm_layer(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                norm_layer(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            norm_layer(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups),
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, q, k, v, xyz, reference_index, new_xyz=None):
        if new_xyz is None:
            new_xyz = xyz
        
        query, key, value = (
            self.linear_q(q),
            self.linear_k(k),
            self.linear_v(v),
        )

        key = pointops.grouping(reference_index, key, xyz, new_xyz, with_xyz=True)
        value = pointops.grouping(reference_index, value, xyz, new_xyz, with_xyz=False)
        pos, key = key[:, :, 0:3], key[:, :, 3:]
        relation_qk = key - query.unsqueeze(1)
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))

        mask = torch.sign(reference_index + 1)
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
        feat = einops.rearrange(feat, "n g i -> n (g i)")
        return feat


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_channels,
        groups,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        enable_checkpoint=False,
    ):
        super().__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            norm_layer=norm_layer,
        )
        self.mlp = MLP(embed_channels, embed_channels, act_layer=nn.ReLU, drop=0.0)
        self.norm1 = norm_layer(embed_channels)
        self.norm2 = norm_layer(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, points, reference_index):
        coord, feat, offset = points
        identity = feat

        feat = (
            self.attn(feat, feat, feat, coord, reference_index)
            if not self.enable_checkpoint
            else checkpoint(self.attn, feat, feat, feat, coord, reference_index)
        )
        
        feat = self.norm1(feat)
        feat = self.mlp(feat)
        feat = self.norm2(feat)
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset]


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_channels,
        groups,
        use_cross_attn=True,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        enable_checkpoint=False,
    ):
        super().__init__()
        self.self_attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            norm_layer=norm_layer,
        )

        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = GroupedVectorAttention(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                norm_layer=norm_layer,
            )
            self.norm2 = norm_layer(embed_channels)

        self.mlp = MLP(embed_channels, embed_channels, act_layer=nn.ReLU, drop=0.0)
        self.norm1 = norm_layer(embed_channels)
        self.norm3 = norm_layer(embed_channels)

        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, points, reference_index_self_attn, skip_points=None, reference_index_cross_attn=None):
        coord, feat, offset = points
        if self.use_cross_attn and skip_points is not None:
            skip_coord, skip_feat, skip_offset = skip_points

        identity = feat
        
        feat = (
            self.self_attn(feat, feat, feat, coord, reference_index_self_attn)
            if not self.enable_checkpoint
            else checkpoint(self.attn, feat, feat, feat, coord, reference_index_self_attn)
        )
        feat = identity + self.drop_path(self.norm1(feat))
        identity = feat

        if self.use_cross_attn and skip_points is not None:
            feat = (
                self.cross_attn(feat, skip_feat, skip_feat, skip_coord, reference_index_cross_attn, new_xyz=coord)
                if not self.enable_checkpoint
                else checkpoint(self.cross_attn, feat, skip_feat, skip_feat, skip_coord, reference_index_cross_attn, new_xyz=coord)
            )
            feat = self.norm2(feat)
            feat = identity + self.drop_path(feat)
            identity = feat

        feat = self.mlp(feat)
        feat = self.norm3(feat)
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset]


class EncoderBlockSequence(nn.Module):
    def __init__(
        self,
        depth,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        enable_checkpoint=False,
    ):
        super().__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = EncoderBlock(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                norm_layer=norm_layer,
                enable_checkpoint=enable_checkpoint,
            )
            self.blocks.append(block)

    def forward(self, points):
        coord, feat, offset = points
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        assert coord.dtype == torch.float32, "Point coordinates should be float32"
        reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)
        for block in self.blocks:
            points = block(points, reference_index)
        return points


class DecoderBlockSequence(nn.Module):
    def __init__(
        self,
        depth,
        embed_channels,
        groups,
        use_cross_attn=True,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        enable_checkpoint=False,
    ):
        super().__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = DecoderBlock(
                embed_channels=embed_channels,
                groups=groups,
                use_cross_attn=use_cross_attn,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                norm_layer=norm_layer,
                enable_checkpoint=enable_checkpoint,
            )
            self.blocks.append(block)

    def forward(self, points, skip_points=None):
        coord, feat, offset = points
        if skip_points is None:
            skip_coord, skip_offset = coord, offset
            reference_index_cross_attn = None
        else:
            skip_coord, skip_feat, skip_offset = skip_points
            assert skip_coord.dtype == torch.float32, "Point coordinates should be float32"
            reference_index_cross_attn, _ = pointops.knn_query(self.neighbours, 
                skip_coord, skip_offset, coord, offset)
        
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        assert coord.dtype == torch.float32, "Point coordinates should be float32"
        reference_index_self_attn, _ = pointops.knn_query(self.neighbours, coord, offset)
        for block in self.blocks:
            points = block(points, reference_index_self_attn, skip_points, reference_index_cross_attn)
        return points


class Encoder(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        stride=4,
        grid_size=None,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        norm_layer=nn.LayerNorm,
        enable_checkpoint=False,
    ):
        super(Encoder, self).__init__()

        self.down = TransitionDownFPS(
            in_channels=in_channels,
            out_channels=embed_channels,
            stride=stride,
        )
        # self.down = GridPool(
        #     in_channels=in_channels,
        #     out_channels=embed_channels,
        #     grid_size=grid_size,
        # )

        self.blocks = EncoderBlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            norm_layer=norm_layer,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        points, _ = self.down(points)
        return self.blocks(points)


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        groups,
        depth,
        expansion=4,
        use_cross_attn=True,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        norm_layer=nn.LayerNorm,
        enable_checkpoint=False,
    ):
        super(Decoder, self).__init__()

        if not use_cross_attn:
            self.up = TransitionUp(
                in_channels=embed_channels,
                out_channels=in_channels,
                norm_layer=norm_layer,
            )

        self.expand = Expand(
            in_channels=in_channels,
            out_channels=embed_channels,
            expansion=expansion,
            norm_layer=norm_layer,
        )

        self.use_cross_attn = use_cross_attn

        self.blocks = DecoderBlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            use_cross_attn=use_cross_attn,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            norm_layer=norm_layer,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points, skip_points):
        if not self.use_cross_attn:
            points = self.up(points, skip_points) # enable if not using cross attn
        points = self.expand(points) # c=embed_channels
        return self.blocks(points, skip_points)


class GVAPatchEmbed(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        enable_checkpoint=False,
    ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            norm_layer(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = EncoderBlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        coord, feat, offset = points
        feat = self.proj(feat)
        return self.blocks([coord, feat, offset])


class PointTransformerV2_x(nn.Module):
    def __init__(
        self,
        in_channels,
        patch_embed_depth=1,
        patch_embed_channels=32,
        patch_embed_groups=4,
        patch_embed_neighbours=8,
        first_random_downsample=2,
        enc_depths=(1, 1, 1),
        enc_channels=(64, 128, 256),
        enc_stride=(4, 4, 4),
        # grid_sizes=(0.12, 0.24, 0.48),
        enc_groups=(8, 16, 32),
        enc_neighbours=(8, 8, 8),
        dec_depths=(1, 1, 1),
        dec_use_cross_attn=True,
        dec_channels=(32, 64, 128),
        dec_groups=(4, 8, 16),
        dec_neighbours=(8, 8, 8),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        final_expansion=2,
        norm_layer="ln",
        enable_checkpoint=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_stages = len(enc_depths)
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)

        norm_layer = nn.LayerNorm if norm_layer == "ln" else nn.BatchNorm1d

        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            enable_checkpoint=enable_checkpoint,
        )

        self.first_random_downsample = first_random_downsample > 1
        if first_random_downsample > 1:
            self.down1 = RandomDownSample(down_ratio=first_random_downsample)

        enc_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))
        ]
        dec_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))
        ]
        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                stride=enc_stride[i],
                # grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[
                    sum(enc_depths[:i]) : sum(enc_depths[: i + 1])
                ],
                norm_layer=norm_layer,
                enable_checkpoint=enable_checkpoint,
            )
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                expansion=enc_stride[i],
                use_cross_attn=dec_use_cross_attn,
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[
                    sum(dec_depths[:i]) : sum(dec_depths[: i + 1])
                ],
                norm_layer=norm_layer,
                enable_checkpoint=enable_checkpoint,
            )
            self.enc_stages.append(enc)
            self.dec_stages.append(dec)

        self.final_expand = Expand(
            in_channels=dec_channels[0],
            out_channels=dec_channels[0], 
            expansion=final_expansion
        )


    def forward(self, data_dict):
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()

        # a batch of point cloud is a list of coord, feat and offset
        points = [coord, feat, offset]
        points = self.patch_embed(points) # (N0, 3), (N0, C), (B,)

        skips = [points]
        if self.first_random_downsample:
            points = self.down1(points)
        for i in range(self.num_stages):
            points = self.enc_stages[i](points)
            skips.append(points)  # record points info of current stage
        
        points = skips.pop(-1) # unpooling points info in the last enc stage
        for i in reversed(range(self.num_stages)):
            skip_points = skips.pop(-1)
            points = self.dec_stages[i](points, skip_points)

        points = self.final_expand(points)
        # points = self.final_expand([coord, points[1], points[2]])
        return points
