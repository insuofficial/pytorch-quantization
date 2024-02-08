from .utils import create_quantizer
from ..config import QConfig 


class QModule(object):
    _FLOAT_MODULE = None

    def __init__(
        self,
        act_qconfig: QConfig = None,
        weight_qconfig: QConfig = None,
        act_mtype: str = None,
        weight_mtype: str = None,
    ):
        self.quantize = False
        self.calibrate = False
        self.last_calibrate = False

        # weight
        if weight_qconfig is not None:
            self.weight_config = weight_qconfig
            self.weight_quantizer = create_quantizer(
                weight_qconfig.quantizer,
                weight_qconfig.observer,
                weight_qconfig.scheme,
                weight_qconfig.dtype,
                weight_mtype
            )
            self.weight_regularizer = None

        # activation
        if act_qconfig is not None:
            self.act_config = act_qconfig
            self.act_quantizer = create_quantizer(
                act_qconfig.quantizer,
                act_qconfig.observer,
                act_qconfig.scheme,
                act_qconfig.dtype,
                act_mtype
            )
            self.act_regularizer = None

    def forward(self, inputs, weight):
        if self.calibrate:
            self.act_quantizer.observer.update(inputs)
            self.weight_quantizer.observer.update(weight)
            if self.last_calibrate:
                self.act_quantizer.update_quantization_params(inputs)
                self.weight_quantizer.update_quantization_params(inputs)

        if self.quantize:
            if not self.act_regularizer is None:
                self.act_regularizer(inputs, quantizer=self.act_quantizer)
            if not self.weight_regularizer is None:
                self.weight_regularizer(weight, quantizer=self.weight_quantizer)

        if self.quantize:
            inputs = self.act_quantizer(inputs)
            weight = self.weight_quantizer(weight)

        return inputs, weight

    @classmethod
    def from_float(cls, mod, weight_qconfig, act_qconfig):
        raise NotImplementedError

    def assert_from_float(cls, mod):
        assert isinstance(mod, cls._FLOAT_MODULE), \
            " torq.nn." + cls.__name__ + ".from_float only works for " + \
            cls._FLOAT_MODULE.__name__ + " but got:" + str(type(mod))

