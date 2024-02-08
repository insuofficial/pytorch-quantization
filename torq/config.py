class QConfig(object):
    def __init__(
        self,
        quantizer='uniform',
        observer='minmax',
        scheme='per-tensor',
        dtype='int8'
    ):
        self.quantizer = quantizer
        self.observer = observer
        self.scheme = scheme
        self.dtype = dtype


class QScheduler(object):
    def __init__(
        self,
        weight_qconfig: QConfig,
        act_qconfig: QConfig,
        except_modules: list = []
    ):
        self.weight_qconfig = weight_qconfig
        self.act_qconfig = act_qconfig
        self.except_modules = except_modules

    def get_config(self):
        config = {
            'weight_qconfig': self.weight_qconfig,
            'act_qconfig'   : self.act_qconfig
        }
        return config

