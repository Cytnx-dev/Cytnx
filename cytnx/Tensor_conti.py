from .utils import *
from cytnx import *
import numpy as _nppy
"""
@add_method(Storage)
def astype(self, new_type):
    if(self.dtype() == new_type):
        return self

    else:
        return self.astype_different_type(new_type)
"""

@add_method(Tensor)
def to(self, device):
    if(self.device() == device):
        return self

    else:
        return self.to_different_device(device)

@add_method(Tensor)
def astype(self,dtype):
    if(self.dtype() == dtype):
        return self
    else:
        return self.astype_different_dtype(dtype)

@add_method(Tensor)
def contiguous(self):
    if(self.is_contiguous()):
        return self
    else:
        return self.make_contiguous()

@add_method(Tensor)
def numpy(self):
    return _nppy.array(self)
