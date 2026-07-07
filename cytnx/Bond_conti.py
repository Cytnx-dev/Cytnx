from .utils import *
from cytnx import *
# from typing import List
# Use beartype to check the type of arguments
from beartype.typing import List

# All of Bond's Python-side helpers (redirect_, getDegeneracy, group_duplicates)
# have been folded directly into the C++ bindings in pybind/bond_py.cpp; this
# module intentionally has no remaining delegation wrappers.
