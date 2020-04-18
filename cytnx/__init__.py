from .cytnx import *
from .Storage_conti import *
from .Tensor_conti import *
#from .CyTensor_conti import *
import numpy as np

def from_numpy(np_arr):
    tmp = np_arr
    if np_arr.flags['C_CONTIGUOUS'] == False:
        tmp = np.ascontiguousarray(np_arr)
    return cytnx._from_numpy(tmp)



