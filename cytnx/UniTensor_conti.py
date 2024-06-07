from .utils import *
from cytnx import *
# from typing import List
# Use beartype to check the type of arguments
from beartype.typing import List
from functools import singledispatch

## load the submodule from pybind and inject the methods

class Hclass:

    def __init__(self, Helper_class):
        self.c_cHclass = Helper_class

    def exists(self):
        return self.c_cHclass.exists()

    def __getattr__(self, name):

        if name == "value":

            if not self.exists():
                raise ValueError("[ERROR] trying access an element that is not exists!, using T.if_exists = sth or checking with T.exists() to verify before access element!")

            if(self.c_cHclass.dtype()==Type.Double):
                return self.c_cHclass.get_elem_d();
            elif(self.c_cHclass.dtype()==Type.ComplexDouble):
                return self.c_cHclass.get_elem_cd();
            elif(self.c_cHclass.dtype()==Type.Float):
                return self.c_cHclass.get_elem_f();
            elif(self.c_cHclass.dtype()==Type.ComplexFloat):
                return self.c_cHclass.get_elem_f();
            elif(self.c_cHclass.dtype()==Type.Uint64):
                return self.c_cHclass.get_elem_u64();
            elif(self.c_cHclass.dtype()==Type.Int64):
                return self.c_cHclass.get_elem_i64();
            elif(self.c_cHclass.dtype()==Type.Uint32):
                return self.c_cHclass.get_elem_u32();
            elif(self.c_cHclass.dtype()==Type.Int32):
                return self.c_cHclass.get_elem_i32();
            elif(self.c_cHclass.dtype()==Type.Uint16):
                return self.c_cHclass.get_elem_u16();
            elif(self.c_cHclass.dtype()==Type.Int16):
                return self.c_cHclass.get_elem_i16();
            elif(self.c_cHclass.dtype()==Type.Bool):
                return self.c_cHclass.get_elem_b();
            else:
                raise ValueError("[ERROR] invalid type of an element!")

        elif name == "if_exists":
            if(self.exists()):
                if(self.c_cHclass.dtype()==Type.Double):
                    return self.c_cHclass.get_elem_d();
                elif(self.c_cHclass.dtype()==Type.ComplexDouble):
                    return self.c_cHclass.get_elem_cd();
                elif(self.c_cHclass.dtype()==Type.Float):
                    return self.c_cHclass.get_elem_f();
                elif(self.c_cHclass.dtype()==Type.ComplexFloat):
                    return self.c_cHclass.get_elem_f();
                elif(self.c_cHclass.dtype()==Type.Uint64):
                    return self.c_cHclass.get_elem_u64();
                elif(self.c_cHclass.dtype()==Type.Int64):
                    return self.c_cHclass.get_elem_i64();
                elif(self.c_cHclass.dtype()==Type.Uint32):
                    return self.c_cHclass.get_elem_u32();
                elif(self.c_cHclass.dtype()==Type.Int32):
                    return self.c_cHclass.get_elem_i32();
                elif(self.c_cHclass.dtype()==Type.Uint16):
                    return self.c_cHclass.get_elem_u16();
                elif(self.c_cHclass.dtype()==Type.Int16):
                    return self.c_cHclass.get_elem_i16();
                elif(self.c_cHclass.dtype()==Type.Bool):
                    return self.c_cHclass.get_elem_b();
                else:
                    raise ValueError("[ERROR] invalid type of an element!")
            else:
                return None;

        else:
            raise AttributeError("invalid member! %s"%(name))

    def __setattr__(self, name, value):

        if name == "c_cHclass":
            self.__dict__[name] = value
        else:
            if name == "value":
                if not self.exists():
                    raise ValueError("[ERROR] trying access an element that is not exists!, using T.if_exists = sth or checking with T.exists() to verify before access element!")
                self.c_cHclass.set_elem(value);

            elif name == "if_exists":
                if(self.exists()):
                    self.c_cHclass.set_elem(value);

            else:
                raise AttributeError("invalid member! %s"%(name))







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
def Trace_(self,a:int,b:int):
    self.cTrace_(a,b);
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
def normalize_(self):
    self.cnormalize_();
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
def truncate_(self,bond_idx,dim):
    self.ctruncate_(bond_idx,dim);
    return self

@add_method(UniTensor)
def set_name(self,name):
    self.c_set_name(name);
    return self



@add_ovld_method(UniTensor)
def set_label(self, old_label:str, new_label:str):
    self.c_set_label(old_label,new_label);
    return self

@add_ovld_method(UniTensor)
def set_label(self, idx:int, new_label:str):
    self.c_set_label(idx,new_label);
    return self



@add_method(UniTensor)
def set_labels(self,new_labels:List[str]):
    self.c_set_labels(new_labels);
    return self


@add_ovld_method(UniTensor)
def relabel_(self, old_label:str, new_label:str):
    self.c_relabel_(old_label,new_label);
    return self

@add_ovld_method(UniTensor)
def relabel_(self, idx:int, new_label:str):
    self.c_relabel_(idx,new_label);
    return self



@add_ovld_method(UniTensor)
def relabel_(self, old_labels:List[str],new_labels:List[str]):
    self.c_relabel_(old_labels,new_labels);
    return self

@add_ovld_method(UniTensor)
def relabel_(self, new_labels:List[str]):
    self.c_relabel_(new_labels);
    return self

@add_ovld_method(UniTensor)
def relabels_(self, old_labels:List[str],new_labels:List[str]):
    self.c_relabels_(old_labels,new_labels);
    return self

@add_ovld_method(UniTensor)
def relabels_(self, new_labels:List[str]):
    self.c_relabels_(new_labels);
    return self



@add_method(UniTensor)
def set_rowrank_(self,new_rowrank):
    self.c_set_rowrank_(new_rowrank);
    return self



@add_ovld_method(UniTensor)
def at(self, locator:List[int]):
    tmp_hclass = self.c_at(locator);
    return Hclass(tmp_hclass);

@add_ovld_method(UniTensor)
def at(self, labels:List[str], locator:List[int]):
    tmp_hclass = self.c_at(labels,locator);
    return Hclass(tmp_hclass);


@add_method(UniTensor)
def convert_from(self, Tin, force=False):
    self.cfrom(Tin,force);
    return self
