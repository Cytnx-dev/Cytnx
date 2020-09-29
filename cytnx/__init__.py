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
    __cpp_include__=os.path.join(os.path.dirname(os.path.dirname(__file__)),"include")
    __cpp_lib__=os.path.join(os.path.dirname(os.path.dirname(__file__)),"lib")

__blasINTsize__ = cytnx.__blasINTsize__

def from_numpy(np_arr):
    tmp = np_arr
    if np_arr.flags['C_CONTIGUOUS'] == False:
        tmp = numpy.ascontiguousarray(np_arr)
    return cytnx._from_numpy(tmp)

def _find_hptt__():
    hptt_path = None
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),"hptt")):
        hptt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"hptt")
    elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"hptt")):
        hptt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"hptt")
            
    return hptt_path

def _resolve_cpp_compileflags__():
    f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"cxxflags.tmp"))
    lines = f.readlines()
    out = ""
    for line in lines:
        line = line.strip()
        out+=line.replace(";"," ")
        out+=" "
    f.close()
    return out

def _resolve_cpp_linkflags__():
    f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"linkflags.tmp"))
    lines = f.readlines()
    out = ""
    lapack_ldir = ""
    for i in range(len(lines)):
        line = lines[i].strip()
        line = line.replace(";"," ")
        out+=line
        out+=" "
        if i == 0:
            lapack_ldir=os.path.dirname(line.split(' ')[0].strip())
    out += "-Wl,-rpath,%s "%(lapack_ldir)

    hptt_path = _find_hptt__()
    if not hptt_path is None:
        out += os.path.join(hptt_path,"lib/libhptt.a")

    f.close()
    return out


__cpp_linkflags__ = _resolve_cpp_linkflags__()
__cpp_flags__ = _resolve_cpp_compileflags__()

