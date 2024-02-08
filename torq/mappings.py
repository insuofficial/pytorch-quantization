import torch
import timm.models.vision_transformer as timm_vit

from .nn import (
    QLinear, QConv2d, QMatmul,
    vit
)

DEFAULT_MODULE_MAPPINGS = {
    #=== Basic Module ===
    torch.nn.Linear: QLinear,
    torch.nn.Conv2d: QConv2d,
    #=== Model Specific ===
    timm_vit.Attention: vit.QAttention
}
