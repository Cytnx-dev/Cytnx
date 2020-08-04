import os
import numpy
from .cytnx import *
from .Storage_conti import *
from .Tensor_conti import *
from .linalg_conti import *
from .UniTensor_conti import *

__version__ = cytnx.__version__
if(os.path.exists(os.path.join(os.path.dirname(__file__),"include"))):
    # this only set if using anaconda install. 
    __cpp_include__=os.path.join(os.path.dirname(__file__),"include")
    __cpp_lib__=os.path.join(os.path.dirname(__file__),"lib")
else:
    __cpp_include__=''
    __cpp_lib__=''

__blasINTsize__ = cytnx.__blasINTsize__

def from_numpy(np_arr):
    tmp = np_arr
    if np_arr.flags['C_CONTIGUOUS'] == False:
        tmp = numpy.ascontiguousarray(np_arr)
    return cytnx._from_numpy(tmp)

