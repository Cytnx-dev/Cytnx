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
@add_method(CyTensor)
def Conj_(self):
    self.cConj_();
    return self

@add_method(CyTensor)
def Trace_(self,a,b,by_label=False):
    self.cTrace_(a,b,by_label);
    return self

@add_method(CyTensor)
def Transpose_(self):
    self.cTranspose_();
    return self

@add_method(CyTensor)
def Dagger_(self):
    self.cDagger_();
    return self

@add_method(CyTensor)
def tag(self):
    self.ctag();
    return self

@add_method(CyTensor)
def __ipow__(self,p):
    self.c__ipow__(p);
    return self
@add_method(CyTensor)
def Pow_(self,p):
    self.cPow_(p)
    return self

@add_method(CyTensor)
def truncate_(self,bond_idx,dim,by_label=False):
    self.ctruncate_(bond_idx,dim,by_label);
    return self
