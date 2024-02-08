from .base import BaseQuantizer


class IdentityQuantizer(BaseQuantizer):
    def quantize(self, inputs, scale=None, zero_point=None):
        return inputs

    def dequantize(self, inputs, scale=None, zero_point=None):
        return inputs
