import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np

from einops import rearrange
from .resnet_encoder import ResnetEncoder
from ..decoder.resnet_decoder import ResnetDecoder, ResnetDepthDecoder

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

dust3r_path = os.path.join(base_dir, 'submodules/dust3r')

sys.path.append(dust3r_path)

from dust3r.model import AsymmetricCroCo3DStereo

from IPython import embed


class Dust3rEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.pts_feat_dim = 32

        # hardcoding the model name for now
        model_name = "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        self.dust3r = AsymmetricCroCo3DStereo.from_pretrained(f"naver/{model_name}")
        
        print("DUSt3R loaded!")

        self.parameters_to_train = []
        # only train encoder and patch_embed
        self.parameters_to_train += [{"params": self.dust3r.patch_embed.parameters()},
                                     {"params": self.dust3r.enc_blocks.parameters()}]
        
        enc_dim = self.dust3r.enc_embed_dim

        self.use_dust3r_decoder = True
        if self.use_dust3r_decoder:
            self.parameters_to_train += [{"params": self.dust3r.dec_blocks.parameters()}]
            
        else:
            self.proj1 = nn.Linear(enc_dim, self.dust3r.dec_embed_dim)
            self.parameters_to_train += [{"params": self.proj1.parameters()}]

        patch_size = self.dust3r.patch_embed.patch_size
        self.pts_head = nn.Sequential(
            nn.Linear(self.dust3r.dec_embed_dim, 3 * patch_size**2)
        )
        self.pts_feat_dim = nn.Sequential(
            nn.Linear(self.dust3r.dec_embed_dim, self.pts_feat_dim * patch_size**2)
        )
        
    def get_parameter_groups(self):
        return self.parameters_to_train
    
    def forward(self, inputs):
        images = inputs[('color_aug', 0, 0)]
        B, C, H, W = images.shape
        true_shape = inputs.get('true_shape', torch.tensor(images.shape[-2:])[None].repeat(B, 1))

        # vit encoder to get per-image features
        encoded_x, pos, _ = self.dust3r._encode_image(images, true_shape)
        if self.use_dust3r_decoder:
            # get the feature of the final layer of the decoder head
            decoder_outputs, _ = self.dust3r._decoder(encoded_x, pos, encoded_x, pos)
            feats = decoder_outputs[-1]
        else:
            # linear layer to account for the cross-image attention
            feats = self.proj1(encoded_x) # (B, N, dec_embed_dim)
        
        pts = self.pts_head(feats) # (B, N, 3*P^2)
        pts = rearrange(pts, 'b n (d1 d2) -> b n d2 d1', d1=3, d2=self.patch_size**2) # (B, N, p^2, 3)
        pts_feats = self.pts_feat_dim(feats) # (B, N, pts_feat_dim*P^2)
        pts_feats = rearrange(pts_feats, 'b n (d1 d2) -> b n d2 d1', 
            d1=self.pts_feat_dim, d2=self.patch_size**2) # (B, N, p^2, pts_feat_dim)
        pts_feats = torch.cat([pts, pts_feats], dim=-1) # (B, N, p^2, 3+pts_feat_dim)
        pts_feats = rearrange(pts_feats, 'b n d1 d2 -> b (n d1) (d2)') # (B, N*p^2, 3+pts_feat_dim)

        return pts_feats
