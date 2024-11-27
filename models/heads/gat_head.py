import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from collections import OrderedDict
from ..decoder.gaussian_decoder import GaussianDecoder, get_splits_and_inits
from ..heads import head_factory
from .dpt_gs_head import create_gs_head
from IPython import embed

class LinearHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.in_dim = cfg.model.backbone.pts_feat_dim

        self.split_dimensions, scales, biases = get_splits_and_inits(cfg)
        self.num_output_channels = sum(self.split_dimensions)

        self.parameters_to_train = []

        # linear decoder
<<<<<<< HEAD
        self.gaussian_head = nn.Linear(36 + 3, self.num_output_channels)
=======
        self.gaussian_head = nn.Linear(32 + 3, self.num_output_channels)
>>>>>>> [TMP]
        self.parameters_to_train += [{"params": self.gaussian_head.parameters()}]

        # gaussian parameters initialisation
        start_channel = 0
        for out_channel, scale, bias in zip(self.split_dimensions, scales, biases):
            nn.init.xavier_uniform_(
                self.gaussian_head.weight[start_channel:start_channel+out_channel, :], scale)
            nn.init.constant_(
                self.gaussian_head.bias[start_channel:start_channel+out_channel], bias)
            start_channel += out_channel

        # gaussian parameters activation
        self.gaussian_decoder = GaussianDecoder(cfg)

    def get_parameter_groups(self):
        return self.parameters_to_train
    
    def forward(self, pts_with_feats):
        gaussian_params = self.gaussian_head(pts_with_feats)

        # n is the number of gaussians, d is the dimension of the gaussian parameters
        # we reshape to make flash3d's gaussian decoder happy
        gaussian_params = rearrange(gaussian_params, 'b n d -> b d n')
        out = self.gaussian_decoder(gaussian_params, self.split_dimensions)

        return out
