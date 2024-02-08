from .base import BaseQuantizer


class Log2Quantizer(BaseQuantizer):
    def update_quantization_params(self, *args, **kwargs):
        params = self.observer.get_quantization_params(*args, **kwargs)
        self.scale, self.zero_point = params
        self.scale = self.scale.log2().round()

    def quantize(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = inputs / scale + zero_point
        outputs.round_()
        outputs.clamp_(self.dtype.lower_bound, self.dtype.upper_bound)
        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = (inputs - zero_point) * scale
        return outputs
