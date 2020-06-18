import numpy
from .cytnx import *
from .Storage_conti import *
from .Tensor_conti import *
from .linalg_conti import *
#from .CyTensor_conti import *

def from_numpy(np_arr):
    tmp = np_arr
    if np_arr.flags['C_CONTIGUOUS'] == False:
        tmp = numpy.ascontiguousarray(np_arr)
    return cytnx._from_numpy(tmp)

