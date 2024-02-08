class DType(object):
    def __init__(
        self,
        bits: int,
        signed: bool,
        upper_bound: int,
        lower_bound: int,
        name=None
    ):
        self.bits = bits
        self.signed = signed
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.range = 2**bits
        self.name = name


class INT(DType):
    def __init__(self, bits):
        upper_bound = 2**(bits-1) - 1
        lower_bound = -2**(bits-1)
        name = "int{}".format(bits)
        super().__init__(bits, True, upper_bound, lower_bound, name)


class UINT(DType):
    def __init__(self, bits):
        upper_bound = 2**bits - 1
        lower_bound = 0
        name = "uint{}".format(bits)
        super().__init__(bits, False, upper_bound, lower_bound, name)


DTYPE_DICT = {
    **{'int{}'.format(i):INT(i) for i in range(1, 17)},
    **{'uint{}'.format(i):UINT(i) for i in range(1, 17)}
}


