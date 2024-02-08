import torch


eps = torch.finfo(torch.float32).eps 


class BaseObserver:
    def __init__(self, scheme, dtype, mtype):
        self.scheme = scheme
        self.dtype = dtype
        self.mtype = mtype
        self.max_val = None
        self.min_val = None

    def reshape_tensor(self, v):
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        v = v.detach()
        if self.mtype in ['conv_weight', 'linear_weight']:
            v = v.reshape(v.shape[0], -1)
        elif self.mtype == 'activation':
            if len(v.shape) == 4:
                v = v.permute(0, 2, 3, 1)
            v = v.reshape(-1, v.shape[-1])
            v = v.transpose(0, 1)
        else:
            raise NotImplementedError
        return v

    def update(self, v):
        # update self.max_val and self.min_val
        raise NotImplementedError

    def get_quantization_params(self, *args, **kwargs):
        raise NotImplementedError
