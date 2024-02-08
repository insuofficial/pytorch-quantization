import numpy as np
import torch

from .base import eps
from .base import BaseObserver


class PercentileObserver(BaseObserver):
    def __init__(self, scheme, dtype, mtype, sigma=0.01, alpha=0.99999):
        super().__init__(scheme, dtype, mtype)
        self.sigma = sigma
        self.alpha = alpha

    def update(self, v):
        # support only per-tensor scheme.
        assert self.scheme == 'per-tensor'

        v = self.reshape_tensor(v)
        try:
            cur_max = torch.quantile(v.reshape(-1), self.alpha)
            cur_min = torch.quantile(v.reshape(-1), 1.0-self.alpha)
        except:
            cur_max = torch.tensor(
                np.percentile(v.reshape(-1).cpu(), self.alpha*100),
                device=v.device,
                dtype=torch.float32
            )
            cur_min = torch.tensor(
                np.percentile(v.reshape(-1).cpu(), (1-self.alpha)*100),
                device=v.device,
                dtype=torch.float32
            )
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = self.max_val + self.sigma * (cur_max - self.max_val)
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = self.min_val + self.sigma * (cur_min - self.min_val)

    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.dtype.upper_bound
        qmin = self.dtype.lower_bound

        scale = torch.ones_like(max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(max_val, dtype=torch.int32)

        if self.dtype.signed: # symmetric
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale.clamp_(eps)
            zero_point = torch.zeros_like(max_val, dtype=torch.int32)
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale.clamp_(eps)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)
        return scale, zero_point
