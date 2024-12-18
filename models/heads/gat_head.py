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
    def __init__(self, cfg, in_dims=[32], xyz_scale=0.01, xyz_bias=0.0):
        super().__init__()

        self.cfg = cfg

        self.split_dimensions, scales, biases = get_splits_and_inits(cfg)
        self.num_output_channels = sum(self.split_dimensions)

        self.parameters_to_train = []

        # Linear decoders for each in_dim
        self.gaussian_heads = nn.ModuleList([
            nn.Linear(in_dim, self.num_output_channels) for in_dim in in_dims
        ])

        for gaussian_head in self.gaussian_heads:
            self.parameters_to_train += [{"params": gaussian_head.parameters()}]

        # Gaussian parameters initialization
        for idx, gaussian_head in enumerate(self.gaussian_heads):
            inv_idx = len(self.gaussian_heads) - idx
            start_channel = 0
            for feat_idx, (out_channel, scale, bias) in enumerate(zip(self.split_dimensions, scales, biases)):
                # by default, the init for scale is the second one
                if feat_idx == 1:
                    bias += np.log(10 * inv_idx)
                nn.init.xavier_uniform_(
                    gaussian_head.weight[start_channel:start_channel + out_channel, :], scale)
                nn.init.constant_(
                    gaussian_head.bias[start_channel:start_channel + out_channel], bias)
                start_channel += out_channel

        if self.cfg.model.predict_offset:
            # Linear offset decoders for each in_dim
            self.offset_heads = nn.ModuleList([
                nn.Linear(in_dim, 3) for in_dim in in_dims
            ])
            for offset_head in self.offset_heads:
                self.parameters_to_train += [{"params": offset_head.parameters()}]

            # Offset parameters initialization
            for offset_head in self.offset_heads:
                nn.init.xavier_uniform_(offset_head.weight[:, :], xyz_scale)
                nn.init.constant_(offset_head.bias[:], xyz_bias)

        self.gaussian_decoder = GaussianDecoder(cfg)
        self.parameters_to_train += [{"params": self.gaussian_decoder.parameters()}]


    def get_parameter_groups(self):
        return self.parameters_to_train
    
    def forward(self, feats):
        integrated_gaussian_params = []
        integrated_gauss_offset = []

        for i, feat in enumerate(feats):
            # breakpoint()
            gaussian_params = self.gaussian_heads[i](feat) 
            gaussian_params = rearrange(gaussian_params, 'b n d -> b d n')
            integrated_gaussian_params.append(gaussian_params)

            # Extra MLP for offset prediction
            if self.cfg.model.predict_offset:
                pts_offsets = self.offset_heads[i](feat) 
                pts_offsets = rearrange(
                    pts_offsets, 'b (s n) d -> b s d n', s=self.cfg.model.gaussians_per_pixel
                )
                integrated_gauss_offset.append(pts_offsets)

        integrated_gaussian_params = torch.cat(integrated_gaussian_params, dim=-1)
        out = self.gaussian_decoder(integrated_gaussian_params, self.split_dimensions)

        if self.cfg.model.predict_offset:
            integrated_gauss_offset = torch.cat(integrated_gauss_offset, dim=-1)
            out["gauss_offset"] = integrated_gauss_offset

        return out
