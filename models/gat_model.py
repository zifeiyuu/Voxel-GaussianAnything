import torch
import logging
import time
import torch.nn as nn

from pathlib import Path
from einops import rearrange

from .encoder.layers import BackprojectDepth
from .encoder.dust3r_encoder import Dust3rEncoder
from .encoder.rgb_unidepth_encoder import Rgb_unidepth_Encoder
from .decoder.gauss_util import focal2fov, getProjectionMatrix, K_to_NDC_pp, render_predicted
from .base_model import BaseModel
from .heads.gat_head import LinearHead, ConvHead
from .heads import head_factory
from misc.util import add_source_frame_id
from misc.depth import estimate_depth_scale, estimate_depth_scale_ransac

from .decoder.pointcloud_decoder import PointTransformerDecoder

from IPython import embed
from models.decoder.resnet_decoder import ResnetDecoder

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
        self.use_conv_head = cfg.model.backbone.use_conv_head

        self.parameters_to_train = []

        # define the model
        if "dust3r" in cfg.model.backbone.name:
            self.encoder = Dust3rEncoder(cfg)
        elif "unidepth" in cfg.model.backbone.name:
            self.encoder = Rgb_unidepth_Encoder(cfg)
            
        self.parameters_to_train += self.encoder.get_parameter_groups()

        self.use_decoder_3d = cfg.model.use_decoder_3d
        if self.use_decoder_3d:
            self.decoder_3d = PointTransformerDecoder(cfg)
            self.parameters_to_train += self.decoder_3d.get_parameter_groups()

        if self.use_conv_head:
            self.decoder_gs = ConvHead(cfg, self.encoder.unidepth.pixel_encoder.n_blocks, self.encoder.enc_dim)
        else:
            self.decoder_gs = LinearHead(cfg)
        self.parameters_to_train += self.decoder_gs.get_parameter_groups()


    def forward(self, inputs):
        cfg = self.cfg
        
        # we predict points and associated features in 3d space directly
        # we do not use unprojection, so as camera intrinsics

        pts3d, original_encoder_outputs, encoder_outputs, pts_feat, pts_rgb = self.encoder(inputs) # (B, N, 3) and (B, N, C)

        B, C, H, W = inputs["color_aug", 0, 0].shape

        if self.use_decoder_3d:
            pts3d, pts_feat = self.decoder_3d(pts3d, torch.cat([pts_rgb, pts_feat], dim=-1))
            
        # predict gaussian parameters for each point
        pts_feat = rearrange(pts_feat, "b (h w) d -> b h w d", h=H, w=W)
        copy_layer = [encoder_outputs] * len(original_encoder_outputs) #try
        if self.use_conv_head:
            outputs = self.decoder_gs(copy_layer, inputs)
        else:
            outputs = self.decoder_gs(torch.cat([pts_feat, pts3d, pts_rgb], dim=-1))

        # add predicted gaussian centroid offset with pts3d to get the final 3d centroids
        pts3d_reshape = rearrange(pts3d, "b (s n) c -> b s c n", s=cfg.model.gaussians_per_pixel)
        # outputs["gauss_means"] = outputs["gauss_offset"] + pts3d_reshape
        outputs["gauss_means"] = pts3d_reshape[:, :, :3, :]

        if cfg.model.gaussian_rendering:
            self.process_gt_poses(inputs, outputs)
            self.render_images(inputs, outputs)

        return outputs

