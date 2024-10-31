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
sys.path.insert(0, dust3r_path)
from dust3r.model import AsymmetricCroCo3DStereo
sys.path.pop(0)

mast3r_path = os.path.join(base_dir, 'submodules/mast3r')
sys.path.insert(0, mast3r_path)
from mast3r.model import AsymmetricMASt3R
sys.path.pop(0)

from IPython import embed

def freeze_all_params(modules):
    if isinstance(modules, list) or isinstance(modules, tuple):
        for module in modules:
            try:
                for n, param in module.named_parameters():
                    param.requires_grad = False
            except AttributeError:
                # module is directly a parameter
                module.requires_grad = False
    else:
        assert isinstance(modules, nn.Module)
        try:
            for n, param in modules.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            modules.requires_grad = False


class Dust3rEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.pts_feat_dim = cfg.model.backbone.pts_feat_dim
        version = cfg.model.backbone.version

        # hardcoding the model name for now
        if "dust3r" in version:
            model_name = "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
            self.dust3r = AsymmetricCroCo3DStereo.from_pretrained(f"naver/{model_name}")
            print("DUSt3R loaded!")
        elif "mast3r" in version:
            model_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
            self.dust3r = AsymmetricMASt3R.from_pretrained(f"naver/{model_name}")
            print("MASt3R loaded!")
        else:
            raise ValueError(f"Unknown version of Dust3r: {version}")

        self.parameters_to_train = []
        all_dust3r_modules = [
            "patch_embed",
            "enc_blocks",
            "enc_norm",
            "decoder_embed",
            "dec_blocks",
            "dec_blocks2",
            "dec_norm",
            "downstream_head1",
            "downstream_head2"
        ]
        modules_to_delete = ["downstream_head2", "prediction_head", "mask_token"]
        modules_to_freeze = ["patch_embed"]
        
        enc_dim = self.dust3r.enc_embed_dim
        self.patch_size = self.dust3r.patch_embed.patch_size[0]

        self.use_dust3r_decoder = cfg.model.backbone.use_dust3r_decoder
        if not self.use_dust3r_decoder:
            modules_to_delete += [
                "decoder_embed",
                "dec_blocks",
                "dec_blocks2",
                "dec_norm",
                "downstream_head1"
            ]

            self.pts_head = nn.Sequential(
                nn.Linear(enc_dim, 3 * self.patch_size**2)
            )
            self.parameters_to_train += [{"params": list(self.pts_head.parameters())}]

        self.pts_feat_head = nn.Sequential(
            nn.Linear(enc_dim, self.pts_feat_dim * self.patch_size**2)
        )
        self.parameters_to_train += [{"params": list(self.pts_feat_head.parameters())}]

        # freeze ,delete and add modules
        if cfg.model.backbone.freeze_encoder:
            modules_to_freeze += ["enc_blocks", "enc_norm"]
        if cfg.model.backbone.freeze_decoder:
            modules_to_freeze += ["decoder_embed", "dec_blocks", "dec_blocks2", "dec_norm"]
        if cfg.model.backbone.freeze_head:
            modules_to_freeze += ["downstream_head1"]

        for module in all_dust3r_modules:
            if module in modules_to_delete:
                delattr(self.dust3r, module)
            elif module in modules_to_freeze:
                freeze_all_params(getattr(self.dust3r, module))
            else:
                self.parameters_to_train += [{"params": list(getattr(self.dust3r, module).parameters())}]

    def get_parameter_groups(self):
        return self.parameters_to_train
    
    def forward(self, inputs):
        images = inputs[('color_aug', 0, 0)]
        images2 = inputs[('color_aug', 3, 0)]
        B, C, H, W = images.shape
        true_shape = inputs.get('true_shape', torch.tensor(images.shape[-2:])[None].repeat(B, 1))

        # vit encoder to get per-image features
        encoded_x, pos, _ = self.dust3r._encode_image(images, true_shape)
        # encoded_x2, pos2, _ = self.dust3r._encode_image(images2, true_shape)
        if self.use_dust3r_decoder:
            # get the feature of the final layer of the decoder head
            # decoder_outputs1, decoder_outputs2 = self.dust3r._decoder(encoded_x, pos, encoded_x2, pos2)
            decoder_outputs, _ = self.dust3r._decoder(encoded_x, pos, encoded_x, pos)
            outputs = self.dust3r.downstream_head1(decoder_outputs, (true_shape.min(), true_shape.max()))
            pts3d, conf = outputs['pts3d'], outputs['conf'] # (B, H, W, 3) and (B, H, W)
            pts3d = rearrange(pts3d, 'b h w c -> b (h w) c')
            
            pts_feat = self.pts_feat_head(encoded_x) # (B, N, p^2*D)
            pts_feat = rearrange(pts_feat, 'b n (p d) -> b (n p) d', p=self.patch_size**2, d=self.pts_feat_dim)

        else:
            # linear layer to decode
            pts3d = self.pts_head(encoded_x) # (B, N, p^2*3)
            pts3d = rearrange(pts3d, 'b n (p d) -> b (n p) d', p=self.patch_size**2, d=3)
            pts_feat = self.pts_feat_head(encoded_x) # (B, N, p^2*D)
            pts_feat = rearrange(pts_feat, 'b n (p d) -> b (n p) d', p=self.patch_size**2, d=self.pts_feat_dim)

        return pts3d, pts_feat
