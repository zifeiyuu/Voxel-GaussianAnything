import torch
import logging
import time
import torch.nn as nn

from pathlib import Path
from einops import rearrange

from .encoder.layers import BackprojectDepth
from .encoder.dust3r_encoder import Dust3rEncoder
from .decoder.gauss_util import focal2fov, getProjectionMatrix, K_to_NDC_pp, render_predicted
from .base_model import BaseModel
from .heads.gat_head import LinearHead
from misc.util import add_source_frame_id
from misc.depth import estimate_depth_scale, estimate_depth_scale_ransac

from IPython import embed

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
        # checking height and width are multiples of 32
        assert cfg.dataset.width % 32 == 0 and cfg.dataset.height % 32 == 0, "'width' and 'height' must be a multiple of 32"

        self.parameters_to_train = []

        # define the model
        if "dust3r" in cfg.model.backbone.name:
            self.encoder = Dust3rEncoder(cfg)
        self.parameters_to_train += self.encoder.get_parameter_groups()

        self.use_3d_refine = True
        if self.use_3d_refine:
            # TODO: define the point cloud network
            pass
        
        self.gaussian_head = LinearHead(cfg)
        self.parameters_to_train += [{'params': self.gaussian_head.parameters()}]


    def forward(self, inputs):
        cfg = self.cfg
        
        # we predict points and associated features in 3d space directly
        # we do not use unprojection, so as camera intrinsics
        pts3d, pts_feat = self.encoder(inputs) # (B, N, 3) and (B, N, C)

        if self.use_3d_refine:
            # TODO: refine the 3d points and features using point cloud network
            pass

        # predict gaussian parameters for each point
        outputs = self.gaussian_head(torch.cat([pts3d, pts_feat], dim=-1))
        # add predicted gaussian centroid offset with pts3d to get the final 3d centroids
        pts3d_reshape = rearrange(pts3d, "b (s n) c -> b s c n", s=cfg.model.gaussians_per_pixel)
        outputs["gauss_means"] = outputs["gauss_offset"] + pts3d_reshape

        if cfg.model.gaussian_rendering:
            self.process_gt_poses(inputs, outputs)
            self.render_images(inputs, outputs)

        return outputs

