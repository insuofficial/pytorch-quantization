import timm
import torch
import torch.nn as nn

from ..base import QModule
from ..linear import QLinear
from ..matmul import QMatmul


class QAttention(nn.Module, QModule):
    _FLOAT_MODULE = timm.models.vision_transformer.Attention

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.,
        proj_drop=0.,
        norm_layer=nn.LayerNorm,
        weight_qconfig=None,
        act_qconfig=None
    ):
        nn.Module.__init__(self)
        QModule.__init__(self)

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = QLinear(dim, dim * 3, qkv_bias, weight_qconfig, act_qconfig)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn = QMatmul(act_qconfig, act_qconfig)
        self.attn_drop = nn.Dropout(attn_drop)
        self.score = QMatmul(act_qconfig, act_qconfig)
        self.proj = QLinear(dim, dim, True, weight_qconfig, act_qconfig)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        k = k.transpose(-2, -1)
        q = q * self.scale
        # attn = q @ k
        attn = self.attn(q, k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # score = attn @ v
        score = self.score(attn, v)

        x = score.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    # def _get_name(self):
        # return 'QuantizedAttention'

    @classmethod
    def from_float(cls, mod, weight_qconfig, act_qconfig):
        QModule.assert_from_float(cls, mod)
        qmod = cls(
            dim=mod.head_dim * mod.num_heads,
            num_heads=mod.num_heads,
            qkv_bias=mod.qkv.bias is not None,
            qk_norm=mod.q_norm is not None,
            attn_drop=mod.attn_drop.p,
            proj_drop=mod.proj_drop.p,
            norm_layer=type(mod.q_norm),
            weight_qconfig=weight_qconfig,
            act_qconfig=act_qconfig
        )
        qmod.load_state_dict(mod.state_dict(), strict=False)

        return qmod

