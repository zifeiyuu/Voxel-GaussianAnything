import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from collections import OrderedDict
from ..decoder.gaussian_decoder import GaussianDecoder, get_splits_and_inits
from ..heads import head_factory
from .dpt_gs_head import create_gs_head
from IPython import embed
import random

class LinearHead(nn.Module):
    def __init__(self, cfg, in_dim=32):
        super().__init__()

        self.cfg = cfg

        self.split_dimensions, scales, biases = get_splits_and_inits(cfg)
        self.num_output_channels = sum(self.split_dimensions)

        self.parameters_to_train = []

        # linear decoder
        self.gaussian_head = nn.Linear(in_dim, self.num_output_channels) ## improved hard code, might be buggy
        self.parameters_to_train += [{"params": self.gaussian_head.parameters()}]
        # gaussian parameters initialisation
        start_channel = 0
        for out_channel, scale, bias in zip(self.split_dimensions, scales, biases):
            nn.init.xavier_uniform_(
                self.gaussian_head.weight[start_channel:start_channel+out_channel, :], scale)
            nn.init.constant_(
                self.gaussian_head.bias[start_channel:start_channel+out_channel], bias)
            start_channel += out_channel

        if self.cfg.model.predict_offset:
            # linear offset decoder
            self.offset_head = nn.Linear(in_dim, 3)
            self.parameters_to_train += [{"params": self.offset_head.parameters()}]
            # gaussian parameters initialisation
            nn.init.xavier_uniform_(
                self.offset_head.weight[:, :], cfg.model.xyz_scale)
            nn.init.constant_(
                self.offset_head.bias[:], cfg.model.xyz_bias)

        self.gaussian_decoder = GaussianDecoder(cfg)
        self.parameters_to_train += [{"params": self.gaussian_decoder.parameters()}]

    def get_parameter_groups(self):
        return self.parameters_to_train
    
    def forward(self, pts_with_feats):
        gaussian_params = self.gaussian_head(pts_with_feats)
        # n is the number of gaussians, d is the dimension of the gaussian parameters
        # we reshape to make flash3d's gaussian decoder happy
        gaussian_params = rearrange(gaussian_params, 'b n d -> b d n')
        out = self.gaussian_decoder(gaussian_params, self.split_dimensions)

        #extra mlp for offset prediction
        if self.cfg.model.predict_offset:
            pts_offsets = self.offset_head(pts_with_feats)
            pts_offsets = rearrange(pts_offsets, 'b (s n) d -> b s d n', s=self.cfg.model.gaussians_per_pixel)
            out["gauss_offset"] = pts_offsets

        return out
