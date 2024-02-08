import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import QModule


class QLinear(nn.Linear, QModule):
    _FLOAT_MODULE = nn.Linear

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_qconfig=None,
        weight_qconfig=None
    ):
        nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        QModule.__init__(
            self,
            act_qconfig=act_qconfig,
            weight_qconfig=weight_qconfig,
            act_mtype='activation',
            weight_mtype='linear_weight'
        )

    def forward(self, inputs):
        inputs, weight = QModule.forward(self, inputs, self.weight)
        outputs = F.linear(inputs, weight, self.bias)
        return outputs

    # def _get_name(self):
        # return 'QuantizedLinear'

    @classmethod
    def from_float(cls, mod, weight_qconfig, act_qconfig):
        QModule.assert_from_float(cls, mod)
        qmod = cls(
            in_features=mod.in_features,
            out_features=mod.out_features,
            bias=mod.bias is not None,
            weight_qconfig=weight_qconfig,
            act_qconfig=act_qconfig
        )
        qmod.load_state_dict(mod.state_dict(), strict=False)
        return qmod

