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

class TensorIterator:
    def __init__(self,tn):
        self._tn = tn;
        self._index = 0;

    def __next__(self):
        if(self._index < len(self._tn)):

            result =  self._tn[self._index]
            self._index +=1;
            return result

        else:
            raise StopIteration
@add_method(Tensor)
def __iter__(self):
    return TensorIterator(self)



##=======================
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
def __iadd__(self,right):
    self.c__iadd__(right);
    return self

@add_method(Tensor)
def __isub__(self,right):
    self.c__isub__(right);
    return self

@add_method(Tensor)
def __imul__(self,right):
    self.c__imul__(right);
    return self

@add_method(Tensor)
def __itruediv__(self,right):
    #print("K")
    self.c__itruediv__(right);
    return self

@add_method(Tensor)
def __ifloordiv__(self,right):
    self.c__ifloordiv__(right);
    return self

@add_method(Tensor)
def __imatmul(self,rhs):
    self.c__imatmul__(rhs);
    return self;


@add_method(Tensor)
def Conj_(self):
    self.cConj_()
    return self
@add_method(Tensor)
def Exp_(self):
    self.cExp_()
    return self
@add_method(Tensor)
def InvM_(self):
    self.cInvM_()
    return self
@add_method(Tensor)
def Inv_(self,clip):
    self.cInv_(clip)
    return self


@add_method(Tensor)
def Abs_(self):
    self.cAbs_()
    return self
@add_method(Tensor)
def Pow_(self,p):
    self.cPow_(p);
    return self

@add_method(Tensor)
def __ipow__(self,right):
    self.c__ipow__(right);
    return self
