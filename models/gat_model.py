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

class GATmodel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.cfg = cfg
        # checking height and width are multiples of 32
        assert cfg.dataset.width % 32 == 0 and cfg.dataset.height % 32 == 0, "'width' and 'height' must be a multiple of 32"

        models = {}
        self.parameters_to_train = []

        # define the model
        # TODO: if "dust3r" in cfg.model.backbone.name:
        # encoder transforms images to 3d point maps with features
        self.encoder = Dust3rEncoder(cfg)
        self.parameters_to_train += self.encoder.get_parameter_groups()

        use_pc_net = True
        if use_pc_net:
            # TODO: define the point cloud network
            pass
        
        # self.gaussian_head = nn.Sequential(


    def forward(self, inputs):
        cfg = self.cfg
        
        pts3d_feats = self.encoder(inputs)

        
        return x
