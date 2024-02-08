from .base import BaseQuantizer
from .identity import IdentityQuantizer
from .uniform import UniformQuantizer
from .log2 import Log2Quantizer
from .lsq import LsqQuantizer


QUANTIZER_DICT = {
    'identity': IdentityQuantizer,
    'uniform' : UniformQuantizer,
    'log2'    : Log2Quantizer,
    'lsq'     : LsqQuantizer
}
