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
from .components.attention import CrossAttention
from einops import rearrange
from .components.pos_embed import interpolate_pos_embed_on_the_fly
from .components.patch_embed import PatchEmbed
from .components.vit_blocks import Block_edm
from climate_learn.utils.dist_functions import F_Identity_B_Broadcast, Grad_Inspect
from climate_learn.utils.fused_attn import FusedAttn
from torch.nn.functional import silu
import torch.nn.functional as F



#----------------------------------------------------------------------------
# noise embedding used in the DDPM++ and ADM architectures.

class NoiseEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class LearnedCompressor(nn.Module):
    """Learned spatial downsampling via strided convolutions.

    Replaces bilinear interpolation with a small CNN that learns
    what spatial information to preserve during compression.

    Architecture: Conv2d(stride=r) -> GroupNorm -> GELU -> Conv2d(1x1)
    Input:  [B, C, H, W]
    Output: [B, C, H//r, W//r]
    """
    def __init__(self, max_channels, compress_ratio, hidden_mult=2):
        super().__init__()
        hidden = max_channels * hidden_mult
        self.net = nn.Sequential(
            nn.Conv2d(max_channels, hidden, kernel_size=compress_ratio + 2,
                      stride=compress_ratio, padding=1),
            nn.GroupNorm(num_groups=min(32, hidden), num_channels=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, max_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class LearnedDecompressor(nn.Module):
    """Learned spatial upsampling via sub-pixel convolution (PixelShuffle).

    Reconstructs full-resolution output from compressed transformer features.
    Uses PixelShuffle for artifact-free upsampling.

    Architecture: Conv2d -> GroupNorm -> GELU -> Conv2d -> PixelShuffle(r)
    Input:  [B, C, H//r, W//r]
    Output: [B, C, H, W]
    """
    def __init__(self, channels, compress_ratio):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(32, channels * 2), num_channels=channels * 2),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels * compress_ratio ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(compress_ratio),
        )

    def forward(self, x):
        return self.net(x)


@register("edm")
class EDM(nn.Module):
    def __init__(
        self,
        default_vars,  #list of default variables to be used for training
        img_size,
        in_channels,
        out_channels,
        history,
        cnn_ratio = 4,
        patch_size=16,
        drop_path=0.1,
        drop_rate=0.1,
        learn_pos_emb=False,
        embed_dim=1024,
        depth=24,
        decoder_depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        tensor_par_size = 1,
        tensor_par_group = None,
        FusedAttn_option = FusedAttn.DEFAULT,
        sigma_data = 0.5,
        compress_ratio = 1,
        use_residual_path = True,
        compress_cond_only = False,
        learned_compression = False,
        keep_ratio = 0.25,
        num_dense_early = 2,
        num_sparse_middle = None,
    ):

        super().__init__()
        self.default_vars = default_vars

        self.use_residual_path = use_residual_path
        self.compress_cond_only = compress_cond_only
        self.learned_compression = learned_compression

        self.img_size = img_size
        self.cnn_ratio = cnn_ratio
        self.in_channels = in_channels   #not actually used. consider deleting it
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.history = history
        self.embed_dim = embed_dim
        self.spatial_resolution = 0
        self.tensor_par_size = tensor_par_size
        self.tensor_par_group = tensor_par_group

        if not 0 < keep_ratio <= 1:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        if not 0 <= num_dense_early <= depth:
            raise ValueError(
                f"num_dense_early must be in [0, {depth}], got {num_dense_early}"
            )
        self.keep_ratio = keep_ratio
        self.num_dense_early = num_dense_early
        self.num_sparse_middle = (
            max(0, depth - 2 * num_dense_early)
            if num_sparse_middle is None
            else num_sparse_middle
        )
        if self.num_sparse_middle < 0:
            raise ValueError("num_sparse_middle must be non-negative")
        self.num_dense_late = depth - self.num_dense_early - self.num_sparse_middle
        if self.num_dense_late < 0:
            raise ValueError(
                "Invalid block split: dense early + sparse middle exceeds depth"
            )


        self.sigma_data = sigma_data



        noise_channels = 256

        self.map_noise = NoiseEmbedding(num_channels=noise_channels, endpoint=True)

        self.map_layer0 = nn.Linear(in_features=noise_channels, out_features=embed_dim)
        self.map_layer1 = nn.Linear(in_features=embed_dim, out_features=embed_dim)


        self.spatial_embed = nn.Linear(1, embed_dim)
        self.temporal_embed = nn.Linear(1, embed_dim)



        self.compress_ratio = compress_ratio

        self.compressed_img_size = tuple((torch.tensor(img_size)//self.compress_ratio).tolist())

        # Token embeddings for conditions (always at compressed resolution)
        self.token_embeds = nn.ModuleList(
            [PatchEmbed(self.compressed_img_size, patch_size, 1, embed_dim) for i in range(len(default_vars))]
        )
        self.num_patches = self.token_embeds[0].num_patches

        # When compress_cond_only=True and compress_ratio>1, cinx stays at full resolution
        # and needs its own token embeddings and patch count
        if self.compress_cond_only and self.compress_ratio > 1:
            self.token_embeds_fullres = nn.ModuleList(
                [PatchEmbed(img_size, patch_size, 1, embed_dim) for i in range(len(default_vars))]
            )
            self.num_patches_fullres = self.token_embeds_fullres[0].num_patches
        else:
            self.token_embeds_fullres = None
            self.num_patches_fullres = self.num_patches

        # Learned compression/decompression modules (replace bilinear interpolation)
        if self.learned_compression and self.compress_ratio > 1:
            # Separate compressors for conditions and cinx (different channel counts)
            self.cond_compressor = LearnedCompressor(in_channels, compress_ratio)
            if not self.compress_cond_only:
                self.cinx_compressor = LearnedCompressor(out_channels, compress_ratio)
                self.decompressor = LearnedDecompressor(out_channels, compress_ratio)
            else:
                self.cinx_compressor = None
                self.decompressor = None
        else:
            self.cond_compressor = None
            self.cinx_compressor = None
            self.decompressor = None

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables

        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        #self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.var_agg = CrossAttention(embed_dim, fused_attn=FusedAttn_option, num_heads=num_heads, qkv_bias=False,tensor_par_size = tensor_par_size, tensor_par_group = tensor_par_group)

        # temporal aggregation: a learnable query and a single-layer cross attention
        self.temporal_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        self.temporal_agg = CrossAttention(embed_dim, fused_attn=FusedAttn_option, num_heads=num_heads, qkv_bias=False,tensor_par_size = tensor_par_size, tensor_par_group = tensor_par_group)


        # pos_embed is sized for cinx (full-res when compress_cond_only, else compressed)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches_fullres, embed_dim), requires_grad=learn_pos_emb
        )
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_drop2 = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]

        self.blocks = nn.ModuleList(
            [
                Block_edm(
                    embed_dim,
                    num_heads =num_heads,
                    fused_attn=FusedAttn_option,
                    mlp_ratio = mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    proj_drop=drop_rate,
                    attn_drop=drop_rate,
                    tensor_par_size = tensor_par_size,
                    tensor_par_group = tensor_par_group,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        #skip connection path
        self.path2 = nn.ModuleList()
        self.path2.append(nn.Conv2d(in_channels=out_channels, out_channels=cnn_ratio, kernel_size=(3, 3), stride=1, padding=1))
        self.path2.append(nn.GELU())
        self.path2.append(nn.Conv2d(in_channels=cnn_ratio, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1))
        self.path2 = nn.Sequential(*self.path2)



        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        # Determine decoder head projection size:
        # - compress_cond_only: cinx is full-res, so project to patch_size^2
        # - learned_compression: decompressor handles upsampling, so project to patch_size^2
        # - otherwise: decoder must upscale from compressed, so project to (compress_ratio*patch_size)^2
        if (self.compress_cond_only and self.compress_ratio > 1) or self.learned_compression:
            head_patch_size = patch_size
        else:
            head_patch_size = self.compress_ratio * patch_size
        self.head.append(nn.Linear(embed_dim, out_channels * head_patch_size**2))
        self.head = nn.Sequential(*self.head)

        #self.conv_out = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.initialize_weights()

    def initialize_weights(self):
        # pos_embed matches cinx resolution: full-res when compress_cond_only, else compressed
        if self.compress_cond_only and self.compress_ratio > 1:
            pe_h = self.img_size[0] // self.patch_size
            pe_w = self.img_size[1] // self.patch_size
        else:
            pe_h = self.compressed_img_size[0] // self.patch_size
            pe_w = self.compressed_img_size[1] // self.patch_size
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            pe_h,
            pe_w,
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



    def unpatchify(self, x: torch.Tensor, scaling =1, out_channels=1):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = out_channels
        if self.compress_cond_only and self.compress_ratio > 1:
            # cinx was at full resolution, so reconstruct from img_size
            h = self.img_size[0] // p
            w = self.img_size[1] // p
        else:
            h = self.compressed_img_size[0] * scaling // p
            w = self.compressed_img_size[1] * scaling // p
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
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map



    def aggregate_temporal(self, x: torch.Tensor):
        """
        x: B,History (or 1), L, D
        """
        b, h, l, _ = x.shape

        x = torch.einsum("bhld->blhd", x)
        x = x.flatten(0, 1)  # B*L, H, D

        temporal_query = self.temporal_query.expand(x.shape[0], -1, -1).contiguous()  #B*L, 1, D

        x = self.temporal_agg(temporal_query, x)  # B*L, 1 , D, where 1 is the aggregated temporal

        x = x.squeeze()   #B*L, D

        if self.tensor_par_size >1:

            src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
            x= F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)  #must do the backward broadcast because of the randomneess of dropout

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L,  D

        return x




    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B,History (or 1), V, L, D
        """
        b, h, _, l, _ = x.shape

        x = torch.einsum("bhvld->bhlvd", x)
        x = x.flatten(0, 2)  # B*H*L, V, D

        #var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)

        var_query = self.var_query.expand(x.shape[0], -1, -1).contiguous()

        #x , _ = self.var_agg(var_query, x, x)
        x = self.var_agg(var_query, x)  # B*H*L, V~ , D, where V~ is the aggregated variables

        x = x.squeeze()

        if self.tensor_par_size >1:

            src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
            x= F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)  #must do the backward broadcast because of the randomneess of dropout

        x = x.unflatten(dim=0, sizes=(b,h, l))  # B,H, L, V~, D

        return x


    def residual_connection(self, x: torch.Tensor):
        """
         x: B, out_channels, H, W
        """
        path2_result = self.path2(x)
        return path2_result



    def data_config(self, res, img_size, in_channels, out_channels):
        with torch.no_grad():
            self.spatial_resolution = res
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.img_size = img_size
            self.compressed_img_size = tuple((torch.tensor(img_size)//self.compress_ratio).tolist())

            self.num_patches = self.compressed_img_size[0] * self.compressed_img_size[1]// (self.patch_size **2)

            if self.compress_cond_only and self.compress_ratio > 1:
                self.num_patches_fullres = img_size[0] * img_size[1] // (self.patch_size ** 2)
            else:
                self.num_patches_fullres = self.num_patches

        if not dist.is_initialized() or dist.get_rank() == 0:
            print("updated res is ",res,"img_size",img_size,"in_channels",in_channels,"out_channels",out_channels,"num_patches",self.num_patches,"num_patches_fullres",self.num_patches_fullres,flush=True)


        if not dist.is_initialized() or dist.get_rank() == 0:
            print("model.pos_embed.shape",self.pos_embed.shape,flush=True)


    def forward_encoder(self, cinx: torch.Tensor, conditions: torch.Tensor, in_variables,out_variables,noise_labels):
        #conditions shape [batch,history, input variables, image width, image height]
        #cinx shape [batch, 1, output variables, image width, image height]

        B,_, _, _, _ = conditions.shape


        ###########
        ###aggregate conditions input variables ####
        ###########
        #if dist.get_rank()==0:
        #    print("inside forward encoder. conditions.shape",conditions.shape,flush=True)

        if isinstance(in_variables, list):
            in_variables = tuple(in_variables)

        #tokenize each input variable for conditions separately
        embeds = []
        var_ids = self.get_var_ids(in_variables, conditions.device)

        conditions = conditions.view(B * self.history, len(in_variables), self.compressed_img_size[0], self.compressed_img_size[1])

        for i in range(len(var_ids)):
            id = var_ids[i]
            #if dist.get_rank()==0:
            #    print("conditions i is",i,"id is",id,"self.token_embeds[id](conditions[:,:, i : i + 1]).shape",self.token_embeds[id](conditions[:, i : i + 1]).shape,flush=True)

            embeds.append(self.token_embeds[id](conditions[:, i : i + 1]))
        conditions = torch.stack(embeds, dim=1)  # B*history, input_V, L, D


        #if dist.get_rank()==0:
        #    print("After stacking conditions.shape",conditions.shape,flush=True)


        conditions = conditions.view(B, self.history, len(in_variables), self.num_patches, self.embed_dim)

        #if dist.get_rank()==0:
        #    print("after view conditions.shape",conditions.shape,flush=True)

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, in_variables) #[1,input_variables,D]

        var_embed = var_embed.unsqueeze(1)   #[1,1,input_variables,D]
        var_embed = var_embed.unsqueeze(3)   #[1,1,input_variables,1,D]

        #if dist.get_rank()==0:
        #    print("var_embed.shape",var_embed.shape,flush=True)



        conditions = conditions + var_embed  # B,History, V, L, D



        #if dist.get_rank()==0:
        #    print("after add var_embed unsqueeze. conditions.shape",conditions.shape,flush=True)


        # variable aggregation
        conditions = self.aggregate_variables(conditions)  # B,History, L, D,


        #if dist.get_rank()==0:
        #    print("after variable aggregation. conditions.shape",conditions.shape,flush=True)



        ############
        #add temporal embedding to conditions
        #use linear for now.  later change to fourier temporal embeding
        ############

        time_steps = torch.arange(-self.history, 1, device=conditions.device, dtype=conditions.dtype)
        #if dist.get_rank()==0:
        #    print("time_steps are",time_steps,"time_steps[:self.history]",time_steps[:self.history],flush=True)

        temporal_emb = self.temporal_embed(time_steps[:self.history, None])  # [H, D]


        temporal_emb = temporal_emb.unsqueeze(0).unsqueeze(2)  #1,history,1, D


        #if dist.get_rank()==0:
        #    print("temporal_emb shape for conditions",temporal_emb.shape,flush=True)


        conditions = conditions + temporal_emb  # B, history, L, D


        #if dist.get_rank()==0:
        #    print("conditions.shape after adding temporal embedding",conditions.shape,flush=True)


        ###########################
        #concatenate the temporal history dimension
        ###########################

        conditions = self.aggregate_temporal(conditions)  # [B,History, L, D] -> [B,  L, D]

        conditions = conditions.unsqueeze(dim=1)  #[B,1,L,D]


        # move time to last dim
        #conditions = conditions.permute(0, 2, 3, 1)  # [B, L, D, T]

        # temporal aggregation
        #conditions = self.temporal_proj(conditions)  # [B, L, D, 1]


        # move time dimension back
        #conditions = conditions.permute(0, 3, 1, 2)  # [B,1, L, D]



        ###########
        ###aggregate cinx noisy input channels ( out_variables) ####
        #cinx shape [batch, 1, out_variables, img_H, img_W]
        ###########

        #if dist.get_rank()==0:
        #    print("inside forward encoder. cinx.shape",cinx.shape,flush=True)

        if isinstance(out_variables, list):
            out_variables = tuple(out_variables)

        #tokenize each output variable for cinx separately
        embeds = []
        var_ids = self.get_var_ids(out_variables, cinx.device)

        cinx=cinx.squeeze(dim=1)

        # Use full-res token embeddings for cinx when compress_cond_only
        cinx_token_embeds = self.token_embeds_fullres if (self.compress_cond_only and self.compress_ratio > 1) else self.token_embeds

        for i in range(len(var_ids)):
            id = var_ids[i]
            embeds.append(cinx_token_embeds[id](cinx[:, i : i + 1]))
        cinx = torch.stack(embeds, dim=1)  # B, out_V, L, D


        #if dist.get_rank()==0:
        #    print("After stacking cinx.shape",cinx.shape,flush=True)

        cinx = cinx.unsqueeze(1)   #B, 1, out_V, L, D

        #if dist.get_rank()==0:
        #    print("after unsqueeze cinx.shape",cinx.shape,flush=True)

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, out_variables) #[1,out_variables,D]

        var_embed = var_embed.unsqueeze(1)   #[1,1,out_variables,D]
        var_embed = var_embed.unsqueeze(3)   #[1,1,out_variables,1,D]

        #if dist.get_rank()==0:
        #    print("var_embed.shape",var_embed.shape,flush=True)



        cinx = cinx + var_embed  # B,1, V, L, D



        #if dist.get_rank()==0:
        #    print("after add var_embed unsqueeze. cinx.shape",cinx.shape,flush=True)


        # variable aggregation
        cinx = self.aggregate_variables(cinx)  # B,1, L, D,


        #if dist.get_rank()==0:
        #    print("after variable aggregation. cinx.shape",cinx.shape,flush=True)

        ######################
        #add temporal emb for cinx
        ######################

        #if dist.get_rank()==0:
        #    print("time_steps are",time_steps,"time_steps[-1]",time_steps[-1],flush=True)

        temporal_emb = self.temporal_embed(time_steps[-1].reshape(1,1)).unsqueeze(0).unsqueeze(2)  # [1,1,1, D]


        #if dist.get_rank()==0:
        #    print("temporal_emb shape for cinx",temporal_emb.shape,flush=True)


        cinx = cinx + temporal_emb  # B, 1, L, D


        #if dist.get_rank()==0:
        #    print("cinx.shape after adding temporal embedding",cinx.shape,flush=True)





        ##############
        #embed noise#
        ##############

        #if dist.get_rank()==0:
        #    print(" noise_labels.shape",noise_labels.shape,flush=True)

        noise_emb = self.map_noise(noise_labels)



        #if dist.get_rank()==0:
        #    print("after map_noise  noise_emb.shape",noise_emb.shape,flush=True)


        noise_emb = silu(self.map_layer0(noise_emb))
        noise_emb = silu(self.map_layer1(noise_emb))


        #if dist.get_rank()==0:
        #    print("after silu noise_emb.shape",noise_emb.shape,flush=True)



        ############
        #concatenate cinx and condition together
        ############
        #aug = torch.cat([cinx, conditions], dim=1)  #B,2, L, D


        #if dist.get_rank()==0:
        #    print("after concat aug.shape",aug.shape,flush=True)




        # aug.shape = [B,2, num_patches,embed_dim]



        ############
        #add spatial positional embedding
        ############

        if self.compress_cond_only and self.compress_ratio > 1:
            # cinx at full resolution, conditions at compressed resolution
            pos_emb_cinx = interpolate_pos_embed_on_the_fly(self.pos_embed, self.patch_size, self.img_size).unsqueeze(0)
            pos_emb_cond = interpolate_pos_embed_on_the_fly(self.pos_embed, self.patch_size, self.compressed_img_size).unsqueeze(0)
            cinx = self.pos_drop1(cinx + pos_emb_cinx)
            conditions = self.pos_drop2(conditions + pos_emb_cond)
        else:
            pos_emb = interpolate_pos_embed_on_the_fly(self.pos_embed, self.patch_size, self.compressed_img_size).unsqueeze(0)
            cinx = self.pos_drop1(cinx + pos_emb)
            conditions = self.pos_drop2(conditions + pos_emb)



        if self.tensor_par_size>1:
            src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
            dist.broadcast(cinx, src_rank , group=self.tensor_par_group)
            dist.broadcast(conditions, src_rank , group=self.tensor_par_group)




        ############
        #add spatial resolution embedding
        ############


        spatial_emb = self.spatial_embed(torch.tensor(self.spatial_resolution,dtype=cinx.dtype,device=cinx.device).unsqueeze(-1))  # D

        spatial_emb = spatial_emb.unsqueeze(0).unsqueeze(0)  #1,1, D

        cinx = cinx.squeeze(dim=1) + spatial_emb  # B, L, D
        conditions = conditions.squeeze(dim=1) + spatial_emb


        #if dist.get_rank()==0:
        #    print("temporal_emb.shape",temporal_emb.shape,"after adding temporal_emb aug.shape",aug.shape,flush=True)


        ###########
        # No folding. Dimension of cinx, cond has dimension of B, 1, L,D
        ###########




        ###########
        # denoising blocks
        ###########


        #if dist.get_rank()==0:
        #    print("after separation cinx.shape",cinx.shape,"cond.shape",cond.shape,flush=True)

        #if dist.get_rank()==0:
        #    print("cinx.dtype",cinx.dtype,"conditions.dtype",conditions.dtype,"noise_emb.dtype",noise_emb.dtype,flush=True)

        block_idx = 0

        # Dense early stage: every spatial token is updated.
        for _ in range(self.num_dense_early):
            cinx = self.blocks[block_idx](cinx, conditions, noise_emb)
            block_idx += 1

        # Sparse middle stage: self-attention runs only on routed queries, while
        # cross-attention keeps the full conditioning sequence as keys/values.
        if self.num_sparse_middle > 0:
            batch_size, sequence_length, hidden_dim = cinx.shape
            num_keep = max(1, int(sequence_length * self.keep_ratio))
            routed_indices = torch.stack(
                [
                    torch.randperm(sequence_length, device=cinx.device)[:num_keep]
                    for _ in range(batch_size)
                ]
            )
            routed_indices, _ = routed_indices.sort(dim=1)
            gather_index = routed_indices.unsqueeze(-1).expand(
                -1, -1, hidden_dim
            )
            routed_tokens = torch.gather(cinx, 1, gather_index)
            routed_input = routed_tokens

            for _ in range(self.num_sparse_middle):
                routed_tokens = self.blocks[block_idx](
                    routed_tokens, conditions, noise_emb
                )
                block_idx += 1

            routed_delta = routed_tokens - routed_input
            delta = torch.zeros_like(cinx)
            delta.scatter_(1, gather_index, routed_delta)
            cinx = cinx + delta

        # Dense late stage restores global token interaction before decoding.
        for _ in range(self.num_dense_late):
            cinx = self.blocks[block_idx](cinx, conditions, noise_emb)
            block_idx += 1
        # cinx.shape = [B,num_patches,embed_dim]
        cinx = self.norm(cinx).to(noise_emb.dtype)

        if self.tensor_par_size>1:
            aug= F_Identity_B_Broadcast(cinx, src_rank, group=self.tensor_par_group)


        #if dist.get_rank()==0:
        #    print("before the end of forward_encoder, aug.shape",aug.shape,flush=True)

        return cinx


#    def find_var_index(self,in_variables,out_variables):
#        temp_index= [in_variables.index(variable) for variable in out_variables]
#        temp_index.append(in_variables.index("land_sea_mask"))
#        temp_index.append(in_variables.index("orography"))
#        temp_index.append(in_variables.index("lattitude"))
#        temp_index.append(in_variables.index("landcover"))


#        return temp_index

    def forward(self, x, in_variables, out_variables, sigma, conditions):
        # noisy input x.shape = [B,1,out_variables,H,W]
        # conditions.shape = [B,History,in_variables,H,W]


        #if dist.get_rank()==0:
        #    print("Inside EDM forward start. x.shape is",x.shape,"sigma.shape before reshape",sigma.shape,flush=True)

        #    print("in_variables is",in_variables,"self.in_channels",self.in_channels,"out_variables is",out_variables,"self.out_channels",self.out_channels,flush=True)

        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        cinx = c_in * x




        #if dist.get_rank()==0:
        #    print("before concat x.shape",x.shape,"conditions.shape",conditions.shape,flush=True)
        #    print("After reshape sigma.shape",sigma.shape,"c_skip.shape",c_skip.shape,"c_out.shape",c_out.shape,"c_in.shape",c_in.shape,"c_noise.shape",c_noise.shape,"cinx.shape",cinx.shape,flush=True)




        ## ---- conv baseline from cinx  ----
        if self.use_residual_path:
            path2_result = self.residual_connection(cinx.squeeze(dim=1))


        #compress data
        B = cinx.size(dim=0)

        if self.compress_cond_only and self.compress_ratio > 1:
            # Only compress conditions; keep cinx at full resolution
            cinx = cinx  # stays at [B, 1, C_out, H, W]
        else:
            # Compress cinx
            if self.learned_compression and self.cinx_compressor is not None:
                cinx = self.cinx_compressor(cinx.squeeze(dim=1))  # [B, C_out, H//r, W//r]
            else:
                cinx = F.interpolate(cinx.squeeze(dim=1), size=(self.compressed_img_size[0],self.compressed_img_size[1]), mode='bilinear', align_corners=False, antialias=True)
            cinx = cinx.unsqueeze(dim=1)

        # Compress conditions
        conditions = conditions.view(B * self.history, len(in_variables), self.img_size[0], self.img_size[1])
        if self.learned_compression and self.cond_compressor is not None:
            conditions = self.cond_compressor(conditions)  # [B*H, C_in, H//r, W//r]
        else:
            conditions = F.interpolate(conditions, size=(self.compressed_img_size[0],self.compressed_img_size[1]), mode='bilinear', align_corners=False, antialias=True)
        conditions = conditions.view(B, self.history, len(in_variables),self.compressed_img_size[0], self.compressed_img_size[1])



        ############
        #forward encoding
        ############
        F_x = self.forward_encoder(cinx, conditions, in_variables,out_variables,c_noise.flatten())

        # x.shape is now [B,num_patches,embed_dim]


        ############
        #decoder. Project back to image
        ############
        F_x = self.head(F_x)




        #if dist.get_rank()==0:
        #    print("after self.head. F_x.shape",F_x.shape,flush=True)


        # F_x.shape is now [B,num_patches,out_channels*patch_size*patch_size]

        # Reconstruct image from patch tokens
        if self.compress_cond_only and self.compress_ratio > 1:
            # cinx was full-res: unpatchify at full resolution
            F_x = self.unpatchify(F_x, scaling=1, out_channels=self.out_channels)
        elif self.learned_compression and self.decompressor is not None:
            # Learned compression: unpatchify at compressed resolution, then learned upsample
            F_x = self.unpatchify(F_x, scaling=1, out_channels=self.out_channels)
            F_x = self.decompressor(F_x)  # [B, C_out, H, W]
        else:
            # Bilinear: unpatchify handles upscaling via larger patch projection
            F_x = self.unpatchify(F_x, scaling=self.compress_ratio, out_channels=self.out_channels)
        # F_x.shape  [B,out_channels,h*patch_size, w*patch_size]


        #if dist.get_rank()==0:
        #    print("after unpatchify. F_x.shape",F_x.shape,flush=True)



        #F_x = self.conv_out(F_x)

        if self.use_residual_path:
            F_x = F_x.unsqueeze(1) + path2_result.unsqueeze(dim=1)
        else:
            F_x = F_x.unsqueeze(1)




        #if dist.get_rank()==0:
        #    print("after conv_out and unsqueeze F_x.shape",F_x.shape,"x.shape",x.shape,flush=True)


#        if path2_result.size(dim=2) !=x.size(dim=2) or path2_result.size(dim=3) !=x.size(dim=3):
#            preds = x + path2_result[:,:,0:x.size(dim=2),0:x.size(dim=3)]
#        else:
        #if dist.get_rank()==0:
        #    print("F_x.shape",F_x.shape,"path2_result.unsqueeze(dim=1).shape",path2_result.unsqueeze(dim=1).shape,flush=True)
        D_x = c_skip * x + c_out * F_x

        D_x = D_x.squeeze(dim=1)

        #if dist.get_rank()==0:
        #    print("The final D_x after squeeze shape",D_x.shape,flush=True)



        return D_x
