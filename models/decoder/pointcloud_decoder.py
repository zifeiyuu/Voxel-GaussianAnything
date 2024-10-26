import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from .pointnet_models.PTv1 import PointTransformerV1, PointTransformerV1_26, PointTransformerV1_38, PointTransformerV1_50

class PointTransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        feat_dim = cfg.model.backbone.pts_feat_dim
        self.transformer = PointTransformerV1_26(in_channels=feat_dim)

    def forward(self, pts3d, pts_feat):
        # pts3d: (B, N, 3)
        # pts_feat: (B, N, C)
        B, N, C = pts_feat.shape
        data_dict = {}

        # flatten B and N dimensions to be compatible with PointTransformer
        data_dict["coord"] = rearrange(pts3d, "B N C -> (B N) C").contiguous()
        data_dict["feat"] = rearrange(pts_feat, "B N C -> (B N) C").contiguous()
        data_dict["offset"] = torch.tensor([t.size(0) for t in pts3d], device=pts3d.device, dtype=torch.long)

        # forward through PointTransformer
        pts3d, pts_feat, offsets = self.transformer(data_dict)

        # unflatten B and N dimensions
        pts3d = rearrange(pts3d, "(B N) C -> B N C", B=B, N=N)
        pts_feat = rearrange(pts_feat, "(B N) C -> B N C", B=B, N=N)

        return pts3d, pts_feat
