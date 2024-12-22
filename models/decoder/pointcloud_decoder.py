import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import copy

from .pointnet_models.PTv1 import PointTransformerV1, PointTransformerV1_26, PointTransformerV1_38, PointTransformerV1_50
from .pointnet_models.myPT import PointTransformerV2_x



from IPython import embed

class PointTransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        feat_dim = cfg.model.backbone.pts_feat_dim
        self.version = cfg.model.version

        kw = dict(copy.deepcopy(cfg.model.decoder_3d))
        kw.pop("name")

        if self.version == 'v3':
            from .pointnet_models.PTv3 import PointTransformerV3Model
            self.transformer =  PointTransformerV3Model(**kw)  
        elif self.version == 'v3_padding':
            from .pointnet_models.PTv3_padding import PointTransformerV3Model
            self.transformer =  PointTransformerV3Model(**kw)  
        else:
            self.transformer = PointTransformerV2_x(**kw)
        self.parameters_to_train = [{"params": list(self.transformer.parameters())}]

    def forward(self, pts3d, pts_feat):
        # pts3d: (B, N, 3)
        # pts_feat: (B, N, C)
        B, N, C = pts_feat.shape
        data_dict = {}

        # flatten B and N dimensions to be compatible with PointTransformer
        data_dict["coord"] = rearrange(pts3d, "B N C -> (B N) C").contiguous()
        data_dict["feat"] = rearrange(pts_feat, "B N C -> (B N) C").contiguous()
        offset = torch.tensor([t.size(0) for t in pts3d], device=pts3d.device, dtype=torch.long)
        offset = torch.cumsum(offset, dim=0)
        data_dict["offset"] = offset

        if self.version == 'v3' or self.version == 'v3_padding':
            data_dict['grid_size'] = self.cfg.model.grid_size
            coords, feats, batchs = self.transformer(data_dict)
            pts3d_list = []
            pts_feat_list = []

            for pts3d, pts_feat in zip(coords, feats):
                # Reshape and add to lists
                pts3d = rearrange(pts3d, "(B N) C -> B N C", B=B)
                pts_feat = rearrange(pts_feat, "(B N) C -> B N C", B=B)

                pts3d_list.append(pts3d)
                pts_feat_list.append(pts_feat)

            return pts3d_list, pts_feat_list
        
        else:
        # forward through PointTransformer
            pts3d, pts_feat, offsets = self.transformer(data_dict)
            # unflatten B and N dimensions, note that N may change after PointTransformer
            pts3d = rearrange(pts3d, "(B N) C -> B N C", B=B)
            pts_feat = rearrange(pts_feat, "(B N) C -> B N C", B=B)
            return pts3d, pts_feat

    def get_parameter_groups(self):
        return self.parameters_to_train
