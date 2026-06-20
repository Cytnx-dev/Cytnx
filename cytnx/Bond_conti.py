from .utils import *
from cytnx import *
# from typing import List
# Use beartype to check the type of arguments
from beartype.typing import List, Union

@add_method(Bond)
def redirect_(self):
    self.c_redirect_();
    return self;

# The native pybind11 overload, captured before @add_method below shadows
# Bond.getDegeneracy with the Python wrapper. Calling it directly when
# return_indices is False skips building and returning the indices list that
# c_getDegeneracy_refarg always fills in.
_c_getDegeneracy = Bond.getDegeneracy

@add_method(Bond)
def getDegeneracy(self, qnum: Union[List[int], Qs], return_indices: bool = False):
    if not return_indices:
        return _c_getDegeneracy(self, qnum)
    inds: List[int] = []
    out = self.c_getDegeneracy_refarg(qnum,inds);
    return out, inds;

@add_method(Bond)
def group_duplicates(self):
    mapper = []
    out = self.c_group_duplicates_refarg(mapper);
    return out, mapper;
