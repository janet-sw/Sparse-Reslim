import torch
import torch.nn as nn
from .attention import Attention
from timm.layers import DropPath
from .mlp import Mlp
from typing import Type, Optional
from climate_learn.utils.fused_attn import FusedAttn
from .time_embed import TimeEmbeddings
from torch.nn.functional import silu
from .attention import CrossAttention


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block_edm(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            fused_attn: FusedAttn = FusedAttn.NONE,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
            tensor_par_size = 1,
            tensor_par_group = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        #self.norm5 = norm_layer(dim)
        #self.norm6 = norm_layer(dim)

        self.attn = Attention(
            dim,
            fused_attn=fused_attn,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            tensor_par_size = tensor_par_size,
            tensor_par_group = tensor_par_group,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
            tensor_par_size = tensor_par_size,
            tensor_par_group = tensor_par_group,
        )

        #self.cond_mlp = mlp_layer(
        #    in_features=dim,
        #    hidden_features=dim,
        #    act_layer=act_layer,
        #    bias=proj_bias,
        #    drop=proj_drop,
        #    tensor_par_size = tensor_par_size,
        #    tensor_par_group = tensor_par_group,
        #)


        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.noise_linear = nn.Linear(in_features=dim, out_features=dim*2)
        self.dim = dim

        #self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads= num_heads, batch_first=True)


        self.cross_attn = CrossAttention(dim, fused_attn=fused_attn, num_heads=num_heads, qkv_bias=True,qk_norm=True,tensor_par_size = tensor_par_size, tensor_par_group = tensor_par_group)



        self.sigma_gate = nn.Linear(in_features=dim,out_features=dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, noise_emb: torch.Tensor) -> torch.Tensor:

        #x shape [B, L, D]
        #cond shape [B, L, D]
        #noise_emb shape [B,D]


        #noise modulation (FLM)
        scale,shift = self.noise_linear(noise_emb).split(self.dim,dim=-1)

        x =   self.norm1(x).to(cond.dtype)


        x = x*(1+scale[:,None,:]) +shift[:,None,:]


        #attention
        x = x + self.drop_path1(self.ls1(self.attn(self.norm2(x).to(cond.dtype))))



        #inject condition at each layer
        cond = self.norm3(cond).to(x.dtype)


        cross_out = self.cross_attn(x,cond)

        gate = torch.sigmoid(self.sigma_gate(noise_emb))


#        if torch.distributed.get_rank()==0:
#            print("gate.shape is",gate.shape,"cross_out.shape",cross_out.shape,flush=True)



        x = x+gate[:,None,:]*cross_out

        #mlp
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm4(x).to(cond.dtype))))

        #cond = cond +0.1*self.cond_mlp(self.norm6(cond))


        return x


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            fused_attn: FusedAttn = FusedAttn.NONE,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
            tensor_par_size = 1,
            tensor_par_group = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            fused_attn=fused_attn,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            tensor_par_size = tensor_par_size,
            tensor_par_group = tensor_par_group,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
            tensor_par_size = tensor_par_size,
            tensor_par_group = tensor_par_group,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x
