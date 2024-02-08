from torch import nn


class BaseQuantizer(nn.Module):
    def __init__(self, observer, dtype, mtype):
        super(BaseQuantizer, self).__init__()
        self.observer = observer
        self.dtype = dtype
        self.mtype = mtype

        # Quantization parameters
        self.scale = None
        self.zero_point = None

    def get_reshape_range(self, inputs):
        range_shape = None
        if self.mtype == 'conv_weight':
            range_shape = (-1, 1, 1, 1)
        elif self.mtype == 'linear_weight':
            range_shape = (-1, 1)
        elif self.mtype == 'activation':
            if len(inputs.shape) == 2:
                range_shape = (1, -1)
            elif len(inputs.shape) == 3:
                range_shape = (1, 1, -1)
            elif len(inputs.shape) == 4:
                range_shape = (1, -1, 1, 1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return range_shape

    def update_quantization_params(self, *args, **kwargs):
        params = self.observer.get_quantization_params(*args, **kwargs)
        self.scale, self.zero_point = params

    def quantize(self, inputs, scale=None, zero_point=None):
        raise NotImplementedError

    def dequantize(self, inputs, scale=None, zero_point=None):
        raise NotImplementedError

    def forward(self, inputs):
        outputs = self.quantize(inputs)
        outputs = self.dequantize(outputs)
        return outputs
