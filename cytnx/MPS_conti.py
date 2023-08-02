from .utils import *
from cytnx import *
from .cytnx.tn_algo import *
## load the submodule from pybind and inject the methods

@add_method(MPS)
def Into_Lortho(self):
    self.c_Into_Lortho();
    return self


@add_method(MPS)
def S_mvleft(self):
    self.c_S_mvleft();
    return self;

@add_method(MPS)
def S_mvright(self):
    self.c_S_mvright();
    return self
