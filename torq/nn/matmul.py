import torch
import torch.nn as nn

from .base import QModule


class QMatmul(nn.Module, QModule):
    _FLOAT_MODULE = None

    def __init__(
        self,
        act_qconfig=None,
        weight_qconfig=None
    ):
        nn.Module.__init__(self)
        QModule.__init__(
            self,
            act_qconfig=act_qconfig,
            weight_qconfig=weight_qconfig,
            act_mtype='activation',
            weight_mtype='activation'
        )

    def forward(self, inputs, other):
        inputs, other = QModule.forward(self, inputs, other)
        outputs = torch.matmul(inputs, other)
        return outputs

    # def _get_name(self):
        # return 'QuantizedMatmul'

    @classmethod
    def from_float(cls, mod):
        raise Exception("from_float can not be used for QMatmul")

