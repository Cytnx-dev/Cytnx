from ..utils import *
from ..cytnx.cytnx_extension_c import *  
from ..cytnx.cytnx_extension_c import xlinalg
## load the submodule from pybind and inject the methods



"""
@add_method(Storage)
def astype(self, new_type):
    if(self.dtype() == new_type):
        return self

    else:
        return self.astype_different_type(new_type)
"""
@add_method(CyTensor)
def to(self, device):
    if(self.device() == device):
        return self

    else:
        return self.to_different_device(device)

@add_method(CyTensor)
def contiguous(self):
    if(self.is_contiguous()):
        return self

    else:
        return self.make_contiguous()
