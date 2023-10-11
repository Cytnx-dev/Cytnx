from .utils import *
from cytnx import *
from .Symmetry_conti import Qs
# from typing import List
# Use beartype to check the type of arguments
from beartype.typing import List

@add_method(Bond)
def redirect_(self):
    self.c_redirect_();
    return self;

@add_method(Bond)
def getDegeneracy(self, qnum: List[int], return_indices:bool):
    inds = []
    out = self.c_getDegeneracy_refarg(qnum,inds);
    return out, inds;

@add_method(Bond)
def getDegeneracy(self, qnum, return_indices:bool):
    inds = []
    out = self.c_getDegeneracy_refarg(lqnum,inds);
    return out, inds;

@add_method(Bond)
def group_duplicates(self):
    mapper = []
    out = self.c_group_duplicates_refarg(mapper);
    return out, mapper;
