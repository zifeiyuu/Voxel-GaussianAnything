import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_scatter import scatter_max

from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from IPython import embed

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def default_param_group(model):
    return [{'params': model.parameters()}]

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PillarsScatter(nn.Module):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, in_channels, output_shape, in_channels_shrink=2):
        super().__init__()
        self.output_shape = output_shape
        self.nx = output_shape[0]
        self.ny = output_shape[1]
        self.nz = output_shape[2]
        self.in_channels = in_channels
        self.fp16_enabled = False
        self.nz_embed = nn.Parameter(torch.zeros(1, self.nz), requires_grad=True)
        torch.nn.init.normal_(self.nz_embed, std=0.02)
        self.linear = nn.Sequential(nn.Linear(in_channels, self.nz, bias=True), nn.LayerNorm(self.nz), nn.ReLU(inplace=True), nn.Linear(self.nz, self.nz))
        self.norm = nn.LayerNorm(self.nz)

    def forward(self, voxel_features, coors, batch_size=None):
        """Foraward function to scatter features."""
        # TODO: rewrite the function in a batch manner
        # no need to deal with different batch cases
        batch_canvas = []
        batch_binary_canvs = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample

            canvas = torch.zeros(
                self.in_channels,
                self.nz, self.nx, self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)
        
            binary_canvas = torch.zeros(
                self.nz, self.nx, self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)


            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            # Now scatter the blob back to the canvas.
            
            indices[:,1] = indices[:,1].clamp(max=self.nz-1)
            indices[:,2] = indices[:,2].clamp(max=self.nx-1)
            indices[:,3] = indices[:,3].clamp(max=self.ny-1)
            
            canvas[:, indices[:,1], indices[:,2], indices[:,3]] = voxels.permute(1,0)
            binary_canvas[indices[:,1], indices[:,2], indices[:,3]] = 1

            # Append to a list for later stacking.
            batch_canvas.append(canvas)
            batch_binary_canvs.append(binary_canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)
        batch_binary_canvs = torch.stack(batch_binary_canvs, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.nz,
                                         self.nx, self.ny).contiguous().permute(0, 1, 3, 4, 2)
        
        # due to GPU memeory, we chose sum pooling first then apply mlp
        feature_volme = (batch_canvas + self.nz_embed.view(1, 1, 1, 1, -1)).sum(-1).permute(0, 2, 3, 1)
        batch_canvas = self.linear(feature_volme)
        batch_canvas = self.norm(batch_canvas).permute(0, 3, 1, 2)

        return batch_canvas, batch_binary_canvs
    
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)


    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x
    
class ConfidenceMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, elementwise_affine=True)
        self.linear = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x
    
    
class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        out_dim=64,
        out_dim_expand=2
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.out_dim = out_dim

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # # Zero-out adaLN modulation layers in DiT blocks:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(self, x):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        for block in self.blocks:
            x = block(x)                      # (N, T, D)
        x_out = self.final_layer(x)              # (N, T, patch_size ** 2 * out_channels)
        x_out = self.unpatchify(x_out)                   # (N, out_channels, H, W)

        return x_out
    
    
class VoxPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # scatter into a 2d canvs
        self.voxel_size_factor = cfg.model.coarse_voxel_size / cfg.model.voxel_size
        self.dim_expand = cfg.model.dim_expand
        canvs_size = (int(abs(cfg.model.pc_range[0] - cfg.model.pc_range[3]) / cfg.model.coarse_voxel_size), int(abs(cfg.model.pc_range[1] - cfg.model.pc_range[4]) / cfg.model.coarse_voxel_size), int(abs(cfg.model.pc_range[2] - cfg.model.pc_range[5]) / cfg.model.coarse_voxel_size))
        self.pts_middle_encoder = PillarsScatter(in_channels=cfg.model.voxel_feat_dim, output_shape=canvs_size, in_channels_shrink=self.dim_expand)
        self.parameters_to_train = default_param_group(self.pts_middle_encoder)
        
        self.dit_transformer = DiT(input_size=canvs_size[0], hidden_size=384, patch_size=2, num_heads=6, in_channels=canvs_size[-1], depth=12, out_dim=cfg.model.voxel_feat_dim, out_dim_expand=self.dim_expand)
        self.parameters_to_train += default_param_group(self.dit_transformer)
        
        
    def voxel_max_pooling_sparse(self, voxels_features, coors):
        batch = coors[:, 0]
        z = coors[:, 1] // self.voxel_size_factor
        x = coors[:, 2] // self.voxel_size_factor
        y = coors[:, 3] // self.voxel_size_factor
        
        coors_reduced = torch.stack([batch, z, x, y], dim=-1)  # [N, 4]

        unique_coors, inverse_indices = torch.unique(coors_reduced, return_inverse=True, dim=0)

        new_voxels_features, _ = scatter_max(voxels_features, inverse_indices, dim=0)
        new_coors = unique_coors

        return new_voxels_features, new_coors
    
        
        
    def forward(self, vfe_feat, coor, max_pooling=True):
        # first, we perfome sparse max pooling here
        if max_pooling:
            vfe_feat, coor = self.voxel_max_pooling_sparse(vfe_feat, coor)
        
        voxel_feature, binary_voxel = self.pts_middle_encoder(vfe_feat, coor, 1)
        pred_binary_logits = self.dit_transformer(voxel_feature)
        
        return pred_binary_logits, binary_voxel, coor
        
        
    def get_parameter_groups(self):
        return self.parameters_to_train