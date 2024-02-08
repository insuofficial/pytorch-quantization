from ..config import QConfig
from ..dtype import DType, DTYPE_DICT 
from ..observer import BaseObserver, OBSERVER_DICT
from ..quantizer import BaseQuantizer, QUANTIZER_DICT


def create_quantizer(
    quantizer,
    observer,
    scheme,
    dtype,
    mtype
) -> BaseQuantizer:
    # Quantizer
    if isinstance(quantizer, BaseQuantizer):
        return quantizer
    elif quantizer in QUANTIZER_DICT.values():
        quantizer = quantizer
    elif isinstance(quantizer, str) and quantizer in QUANTIZER_DICT:
        quantizer = QUANTIZER_DICT[quantizer]
    else:
        raise ValueError(f"Invalid quantizer: {quantizer}")

    # Dtype
    if isinstance(dtype, DType):
        dtype = dtype
    elif isinstance(dtype, str) and dtype in DTYPE_DICT:
        dtype = DTYPE_DICT[dtype]
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

    # Observer
    if isinstance(observer, BaseObserver):
        observer = observer
    elif observer in OBSERVER_DICT.values():
        observer = observer(scheme, dtype, mtype)
    elif isinstance(observer, str) and observer in OBSERVER_DICT:
        observer = OBSERVER_DICT[observer](scheme, dtype, mtype)
    else:
        raise ValueError(f"Invalid observer: {observer}")

    return quantizer(observer, dtype, mtype)

