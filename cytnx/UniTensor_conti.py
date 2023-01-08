from .utils import *
from cytnx import *
## load the submodule from pybind and inject the methods



@add_method(UniTensor)
def astype(self,dtype):
    if(self.dtype() == dtype):
        return self
    else:
        return self.astype_different_type(dtype)

@add_method(UniTensor)
def to(self, device):
    if(self.device() == device):
        return self

    else:
        return self.to_different_device(device)

@add_method(UniTensor)
def contiguous(self):
    if(self.is_contiguous()):
        return self

    else:
        return self.make_contiguous()
@add_method(UniTensor)
def Conj_(self):
    self.cConj_();
    return self

@add_method(UniTensor)
def Trace_(self,a:int,b:int,by_label=False):
    self.cTrace_(a,b,by_label);
    return self

@add_method(UniTensor)
def Trace_(self,a:str,b:str):
    self.cTrace_(a,b);
    return self


@add_method(UniTensor)
def Transpose_(self):
    self.cTranspose_();
    return self

@add_method(UniTensor)
def Dagger_(self):
    self.cDagger_();
    return self

@add_method(UniTensor)
def tag(self):
    self.ctag();
    return self

@add_method(UniTensor)
def __ipow__(self,p):
    self.c__ipow__(p);
    return self
@add_method(UniTensor)
def Pow_(self,p):
    self.cPow_(p)
    return self

@add_method(UniTensor)
def truncate_(self,bond_idx,dim,by_label=False):
    self.ctruncate_(bond_idx,dim,by_label);
    return self

@add_method(UniTensor)
def set_name(self,name):
    self.c_set_name(name);
    return self

@add_method(UniTensor)
def set_label(self,inx,new_label,by_label=False):
    self.c_set_label(inx,new_label,by_label);
    return self


@add_method(UniTensor)
def set_labels(self,new_labels):
    self.c_set_labels(new_labels);
    return self

@add_method(UniTensor)
def set_rowrank(self,new_rowrank):
    self.c_set_rowrank(new_rowrank);
    return self




