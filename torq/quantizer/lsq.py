import torch
from torch import nn

from .base import BaseQuantizer


class STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad):
        return grad

class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad):
        return ctx.scale * grad, None


class LsqQuantizer(BaseQuantizer):
    def update_quantization_params(self, *args, **kwargs):
        super().update_quantization_params(*args, **kwargs)
        self.scale = nn.Parameter(self.scale, requires_grad=True)

    def quantize(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = scale.clamp(min=1e-6)
        scale = scale.reshape(range_shape)
        scale = self._grad_scale(inputs, scale)
        zero_point = zero_point.reshape(range_shape)
        outputs = inputs / scale + zero_point
        outputs = self._clamp_round(outputs)
        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = scale.clamp(min=1e-6)
        scale = self._grad_scale(inputs, scale)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = (inputs - zero_point) * scale
        return outputs

    def _grad_scale(self, inputs, scale):
        upper = self.dtype.upper_bound
        grad_scale = 1.0/((upper*inputs.numel())**0.5)
        scale = GradScale.apply(scale, grad_scale)
        return scale

    def _clamp_round(self, inputs):
        lower = self.dtype.lower_bound
        upper = self.dtype.upper_bound
        outputs = inputs.clamp(lower, upper)
        outputs = STERound.apply(outputs)
        return outputs

