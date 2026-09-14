from .components.cnn_blocks import PeriodicConv2D
from .components.pos_embed import get_2d_sincos_pos_embed
from .utils import register
import torch
import torch.nn as nn
from functools import lru_cache
import numpy as np
import torch.distributed as dist
# Third party
from timm.models.vision_transformer import trunc_normal_
from .components.attention import VariableMapping_Attention
from einops import rearrange
from .components.pos_embed import interpolate_pos_embed_on_the_fly
from .components.patch_embed import PatchEmbed
from .components.vit_blocks import Block
from climate_learn.utils.dist_functions import F_Identity_B_Broadcast, Grad_Inspect
from climate_learn.utils.fused_attn import FusedAttn
from .adaptive_side_channel import DIAG


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block for CNN parallel path"""
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x = input + self.drop_path(x)
        return x


@register("res_slimvit_adaptive")
class Res_Slim_ViT_Adaptive(nn.Module):
    """
    Res_Slim_ViT with adaptive token sparsity.

    Architecture splits into:
    - Dense early blocks (all tokens)
    - Sparse middle blocks (hard tokens only, easy tokens skip via identity)
    - Dense late blocks (all tokens)
    """
    def __init__(
        self,
        default_vars,  # list of default variables to be used for training
        img_size,
        in_channels,
        out_channels,
        history,
        superres_mag=4,
        cnn_ratio=4,
        patch_size=16,
        drop_path=0.1,
        drop_rate=0.1,
        learn_pos_emb=False,
        embed_dim=1024,
        depth=24,
        decoder_depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        tensor_par_size=1,
        tensor_par_group=None,
        FusedAttn_option=FusedAttn.DEFAULT,
        num_constant_vars=4,  # Default: land_sea_mask, orography, lattitude, landcover
        input_refine_cnn=False,  # Optional 3x3 CNN to refine input after downsampling
        # Adaptive sparsity parameters
        keep_ratio=0.25,  # Fraction of tokens to keep in sparse blocks
        num_dense_early=2,  # Number of dense blocks at start
        num_sparse_middle=None,  # Number of sparse blocks (auto if None)
        cnn_hidden_dim=64,  # Hidden dimension for CNN path
        cnn_num_blocks=3,  # Number of ConvNeXt blocks in CNN path
        difficulty_temp=1.0,  # Temperature for difficulty sigmoid
    ):
        super().__init__()
        self.default_vars = default_vars
        self.num_constant_vars = num_constant_vars
        self.input_refine_cnn = input_refine_cnn

        self.img_size = img_size
        self.cnn_ratio = cnn_ratio
        self.superres_mag = superres_mag
        self.in_channels = in_channels * history
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.history = history
        self.embed_dim = embed_dim
        self.spatial_resolution = 0
        self.tensor_par_size = tensor_par_size
        self.tensor_par_group = tensor_par_group

        # Adaptive sparsity configuration
        self.keep_ratio = keep_ratio
        self.num_dense_early = num_dense_early
        self.num_sparse_middle = num_sparse_middle if num_sparse_middle is not None else max(1, depth - 2 * num_dense_early)
        self.num_dense_late = depth - self.num_dense_early - self.num_sparse_middle
        self.difficulty_temp = difficulty_temp

        assert self.num_dense_early + self.num_sparse_middle + self.num_dense_late == depth, \
            f"Block counts don't add up: {self.num_dense_early} + {self.num_sparse_middle} + {self.num_dense_late} != {depth}"

        self.spatial_embed = nn.Linear(1, embed_dim)

        self.token_embeds = nn.ModuleList(
            [PatchEmbed(img_size, patch_size, 1, embed_dim) for i in range(len(default_vars))]
        )
        self.num_patches = self.token_embeds[0].num_patches

        # variable embedding to denote which variable each token belongs to
        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = VariableMapping_Attention(
            embed_dim, fused_attn=FusedAttn_option, num_heads=num_heads, qkv_bias=False,
            tensor_par_size=tensor_par_size, tensor_par_group=tensor_par_group
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=learn_pos_emb
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads=num_heads,
                    fused_attn=FusedAttn_option,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    proj_drop=drop_rate,
                    attn_drop=drop_rate,
                    tensor_par_size=tensor_par_size,
                    tensor_par_group=tensor_par_group,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Difficulty predictor: predicts which tokens are hard
        # self.difficulty_head = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, 1)
        # )

        # Parallel CNN residual path from the input grid to the decoded forecast.
        self.path2 = nn.ModuleList()
        self.path2.append(nn.Conv2d(
            in_channels=(out_channels + num_constant_vars),
            out_channels=cnn_ratio * superres_mag * superres_mag,
            kernel_size=(3, 3), stride=1, padding=1
        ))
        self.path2.append(nn.GELU())
        self.path2.append(nn.PixelShuffle(superres_mag))
        self.path2.append(nn.Conv2d(
            in_channels=cnn_ratio, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1
        ))
        self.path2 = nn.Sequential(*self.path2)

        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, out_channels * (superres_mag * patch_size) ** 2))
        self.head = nn.Sequential(*self.head)

        self.conv_out = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1
        )

        # Optional input refinement CNN
        if self.input_refine_cnn:
            self.input_refine = nn.Sequential(
                nn.Conv2d(in_channels * history, in_channels * history, kernel_size=3, padding=1),
                nn.GELU(),
            )
        else:
            self.input_refine = nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def data_config(self, res, img_size, in_channels, out_channels):
        with torch.no_grad():
            self.spatial_resolution = res
            self.img_size = img_size
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.num_patches = img_size[0] * img_size[1] // (self.patch_size ** 2)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                "updated res is ", res, "img_size", img_size,
                "in_channels", in_channels, "out_channels", out_channels,
                "num_patches", self.num_patches, flush=True
            )
            print("model.pos_embed.shape", self.pos_embed.shape, flush=True)

    def unpatchify(self, x: torch.Tensor, scaling=1, out_channels=1):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = out_channels
        h = self.img_size[0] * scaling // p
        w = self.img_size[1] * scaling // p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape

        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.var_query.expand(x.shape[0], -1, -1).contiguous()
        x = self.var_agg(var_query, x)  # BxL, 1, D
        x = x.squeeze()

        if self.tensor_par_size > 1:
            src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
            x = F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def residual_connection(self, x: torch.Tensor, out_var_index):
        """
        x: B, in channels, H, W
        """
        x = x[:, out_var_index, :, :]
        path2_result = self.path2(x)
        return path2_result

    def select_hard_tokens(self, x, num_keep):
        """
        Select hard tokens based on difficulty prediction.

        The difficulty head is trained via an auxiliary loss in the training loop.
        We store DETACHED input features in DIAG so the training loop can
        recompute logits with a fresh autograd graph (needed because REENTRANT
        activation checkpointing breaks the graph for tensors stored during
        the checkpointed forward pass).

        Args:
            x: (B, L, D) token features after dense early blocks
            num_keep: number of tokens to keep for sparse middle blocks

        Returns:
            hard_tokens: (B, K, D) selected hard tokens
            indices: (B, K) indices of hard tokens in the original sequence
        """
        B, L, D = x.shape

        # Random selection: no difficulty head overhead.
        # Empirically comparable to learned selection (see low-res ablation).
        indices = torch.stack([
            torch.randperm(L, device=x.device)[:num_keep] for _ in range(B)
        ])  # (B, K)

        # Sort indices for better memory access patterns
        indices, _ = indices.sort(dim=1)

        # Gather selected tokens
        indices_expand = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, K, D)
        hard_tokens = torch.gather(x, 1, indices_expand)  # (B, K, D)

        return hard_tokens, indices

    def merge_tokens(self, hard_tokens, easy_tokens, hard_indices, total_length):
        """
        Merge hard and easy tokens back to original order.

        Args:
            hard_tokens: (B, K, D) processed hard tokens
            easy_tokens: (B, L-K, D) processed easy tokens
            hard_indices: (B, K) indices where hard tokens belong
            total_length: L, total number of tokens

        Returns:
            merged: (B, L, D) merged tokens in original order
        """
        B, K, D = hard_tokens.shape

        # Create output tensor
        merged = torch.zeros(B, total_length, D, dtype=hard_tokens.dtype, device=hard_tokens.device)

        # Place hard tokens
        hard_indices_expand = hard_indices.unsqueeze(-1).expand(-1, -1, D)
        merged.scatter_(1, hard_indices_expand, hard_tokens)

        # Create mask for easy positions
        mask = torch.ones(B, total_length, dtype=torch.bool, device=hard_tokens.device)
        mask.scatter_(1, hard_indices, False)

        # Place easy tokens
        merged[mask] = easy_tokens.reshape(-1, D)

        return merged

    def forward_encoder(self, x: torch.Tensor, variables):
        if isinstance(variables, list):
            variables = tuple(variables)

        # Tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        for i in range(len(var_ids)):
            id = var_ids[i]
            embeds.append(self.token_embeds[id](x[:, i : i + 1]))
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # Add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # Variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        # Add positional embedding
        pos_emb = interpolate_pos_embed_on_the_fly(self.pos_embed, self.patch_size, self.img_size)
        x = x + pos_emb

        # Add spatial resolution embedding
        spatial_emb = self.spatial_embed(
            torch.tensor(self.spatial_resolution, dtype=x.dtype, device=x.device).unsqueeze(-1)
        )
        spatial_emb = spatial_emb.unsqueeze(0).unsqueeze(0)  # 1, 1, D
        x = x + spatial_emb  # B, L, D

        x = self.pos_drop(x)

        if self.tensor_par_size > 1:
            src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
            dist.broadcast(x, src_rank, group=self.tensor_par_group)

        # Process through blocks with adaptive sparsity
        block_idx = 0

        # Early dense blocks (all tokens)
        for i in range(self.num_dense_early):
            x = self.blocks[block_idx](x)
            block_idx += 1

        # Middle sparse blocks (hard tokens only, easy tokens skip)
        # Residual formulation: x_out = x_before + delta
        # delta[hard] = Transformer(hard_tokens) - hard_tokens (learned refinement)
        # delta[easy] = 0 (identity, no processing)
        # This ensures the merge is smooth: worst case is "no change", not a
        # representation mismatch between Transformer-processed and raw tokens.
        if self.num_sparse_middle > 0:
            num_keep = max(1, int(x.shape[1] * self.keep_ratio))

            # Select hard tokens
            hard_tokens, hard_indices = self.select_hard_tokens(x, num_keep)

            # Save pre-routing state
            x_before = x.clone()

            # Process hard tokens through transformer blocks
            hard_tokens_input = hard_tokens.clone()
            for i in range(self.num_sparse_middle):
                hard_tokens = self.blocks[block_idx](hard_tokens)
                block_idx += 1

            # Compute delta for hard tokens only
            hard_delta = hard_tokens - hard_tokens_input  # (B, K, D)

            # Apply delta: scatter hard_delta into a zero tensor, add to x_before
            delta = torch.zeros_like(x_before)
            hard_indices_expand = hard_indices.unsqueeze(-1).expand(-1, -1, x_before.shape[-1])
            delta.scatter_(1, hard_indices_expand, hard_delta)
            x = x_before + delta

        # Late dense blocks (all tokens)
        for i in range(self.num_dense_late):
            x = self.blocks[block_idx](x)
            block_idx += 1

        x = self.norm(x)

        if self.tensor_par_size > 1:
            x = F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)

        return x

    def find_var_index(self, in_variables, out_variables):
        temp_index = [in_variables.index(variable) for variable in out_variables]

        # Add optional constant variables if present
        optional_vars = ["land_sea_mask", "orography", "lattitude", "landcover"]
        for var in optional_vars:
            if var in in_variables:
                temp_index.append(in_variables.index(var))

        return temp_index

    def forward(self, x, in_variables, out_variables):
        if len(x.shape) == 5:  # x.shape = [B, T, in_channels, H, W]
            x = x.flatten(1, 2)
        # x.shape = [B, T*in_channels, H, W]

        # Optional input refinement
        if self.input_refine_cnn:
            x = x + self.input_refine(x)  # residual refinement
        else:
            x = self.input_refine(x)  # identity

        out_var_index = self.find_var_index(in_variables, out_variables)
        path2_result = self.residual_connection(x, out_var_index)

        x = self.forward_encoder(x, in_variables)

        # Decoder
        x = self.head(x)

        # Unpatchify
        x = self.unpatchify(x, scaling=self.superres_mag, out_channels=self.out_channels)
        x = self.conv_out(x)

        # Combine with residual path
        if path2_result.size(dim=2) != x.size(dim=2) or path2_result.size(dim=3) != x.size(dim=3):
            preds = x + path2_result[:, :, 0:x.size(dim=2), 0:x.size(dim=3)]
        else:
            preds = x + path2_result

        return preds
