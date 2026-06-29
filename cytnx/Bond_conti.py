from .utils import *
from cytnx import *
# from typing import List
# Use beartype to check the type of arguments
from beartype.typing import List

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
def getDegeneracy(self, qnum: List[int], return_indices: bool = False):
    """Return the degeneracy associated with the given quantum number(s).

    ``qnum`` may be a list of ints or a ``cytnx.Qs`` object. By default only
    the degeneracy is returned; pass ``return_indices=True`` to also get the
    list of matching block indices as a ``(degeneracy, indices)`` tuple.
    """
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
