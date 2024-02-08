import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import QModule


class QConv2d(nn.Conv2d, QModule):
    _FLOAT_MODULE = nn.Conv2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode = 'zeros',
        act_qconfig=None,
        weight_qconfig=None
    ):
        nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode = padding_mode
        )
        QModule.__init__(
            self,
            act_qconfig=act_qconfig,
            weight_qconfig=weight_qconfig,
            act_mtype='activation',
            weight_mtype='conv_weight'
        )

    def forward(self, inputs):
        inputs, weight = QModule.forward(self, inputs, self.weight)
        return self._conv_forward(inputs, weight, self.bias)

    # def _get_name(self):
        # return 'QuantizedConv2d'

    @classmethod
    def from_float(cls, mod, weight_qconfig, act_qconfig):
        QModule.assert_from_float(cls, mod)
        qmod = cls(
            in_channels=mod.in_channels,
            out_channels=mod.out_channels,
            kernel_size=mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,
            weight_qconfig=weight_qconfig,
            act_qconfig=act_qconfig
        )
        qmod.load_state_dict(mod.state_dict(), strict=False)
        return qmod
