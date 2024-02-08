import copy
import logging
from functools import reduce
from typing import Dict, Optional, Iterable

import torch
from torch import nn

from .nn import QModule
from .config import QScheduler
from .mappings import DEFAULT_MODULE_MAPPINGS


_logger = logging.getLogger('torq.quantize')


def find_modules_to_quantize(
    model: nn.Module,
    qscheduler: QScheduler,
    mapping: Dict[type, type]
) -> Dict[str, QModule]:
    modules_to_quantize = {}
    for name, module in model.named_modules():
        if type(module) in mapping.keys():
            if name in qscheduler.except_modules:
                _logger.info(
                    f"Except module '{name}' is found, skip it")
            else:
                qconfigs = qscheduler.get_config()
                cmodule = mapping[type(module)]
                qmodule = cmodule.from_float(module, **qconfigs)
                modules_to_quantize[name] = qmodule
    return modules_to_quantize

def replace_modules_by_name(
    model: nn.Module,
    modules_to_replace: Dict[str, QModule]
) -> nn.Module:
    for name, module in model.named_modules():
        if name in modules_to_replace:
            attr = name.split(sep='.')
            parent = reduce(getattr, attr[:-1], model)
            parent.add_module(attr[-1], modules_to_replace[name])
    return model

def quantize(
    model: nn.Module,
    qscheduler: QScheduler,
    mapping: Optional[Dict[type, type]] = None,
    inplace: bool = False
):
    if mapping is None:
        mapping = DEFAULT_MODULE_MAPPINGS
    if not inplace:
        model = copy.deepcopy(model)
    modules_to_replace = find_modules_to_quantize(model, qscheduler, mapping)
    replace_modules_by_name(model, modules_to_replace)
    return model

@torch.no_grad()
def calibrate(
    model: nn.Module,
    data: Iterable,
    step: int = 10,
    device=None
) -> None:
    data = [next(iter(data))[0] for _ in range(step)]

    def _setattr(key, val):
        for m in model.modules():
            if isinstance(m, QModule):
                setattr(m, key, val)

    _setattr('calibrate', True)
    for i, sample in enumerate(data):
        sample = sample.to(device, non_blocking=True)
        if i == step-1:
            _setattr('last_calibrate', True)
        model(sample)
    _setattr('calibrate', False)
    _setattr('quantize', True)
