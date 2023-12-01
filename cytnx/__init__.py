import os,sys,warnings
from .cytnx import *

if cytnx.__cytnx_backend__ == "torch":
    import numpy, torch

    from .Symmetry_conti import *
    from .Bond_conti import *

else:

    #1) check if numpy is previous imported, if it is, pop warning:
    if ('numpy' in sys.modules) or ('scipy' in sys.modules):
        warnings.warn("numpy and/or scipy are imported before cytnx. Please make sure it support ILP64.")


    ## [NOTE!!] These part has to execute first before import numpy!
    #set_mkl_ilp64()
    def _init_mkl():
        a = zeros(2)
        b = zeros(2)
        linalg.Dot(a,b)
        return 0
    _init_mkl()


    def get_mkl_interface():
        code = get_mkl_code()
        if code < 0:
            raise Warning("does not compile with mkl.")

        if(code%2):
            return "ilp64"
        else:
            return "lp64"


    import numpy


    from .Storage_conti import *
    from .Tensor_conti import *
    from .linalg_conti import *
    from .UniTensor_conti import *
    from .Network_conti import *
    from .MPS_conti import *
    from .Symmetry_conti import *
    from .Bond_conti import *

    def from_numpy(np_arr):
        tmp = np_arr
        if np_arr.flags['C_CONTIGUOUS'] == False:
            tmp = numpy.ascontiguousarray(np_arr)
        return cytnx._from_numpy(tmp)



if(os.path.exists(os.path.join(os.path.dirname(__file__),"include"))):
    # this only set if using anaconda install.
    __cpp_include__=os.path.join(os.path.dirname(__file__),"include")
    __cpp_lib__=os.path.join(os.path.dirname(__file__),"lib")
    if not os.path.isdir(__cpp_lib__):
        __cpp_lib__=os.path.join(os.path.dirname(__file__),"lib64")

else:
    __cpp_include__=os.path.join(os.path.dirname(os.path.dirname(__file__)),"include")
    __cpp_lib__=os.path.join(os.path.dirname(os.path.dirname(__file__)),"lib")
    if not os.path.isdir(__cpp_lib__):
        __cpp_lib__=os.path.join(os.path.dirname(os.path.dirname(__file__)),"lib64")


__blasINTsize__ = cytnx.__blasINTsize__



def _find_hptt__():
    hptt_path = None
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),"hptt")):
        hptt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"hptt")
    elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"hptt")):
        hptt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"hptt")

    return hptt_path

def _find_cutt__():
    cutt_path = None
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),"cutt")):
        cutt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"cutt")
    elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"cutt")):
        cutt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"cutt")

    return cutt_path


def _get_version__():
    f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"version.tmp"))
    line = f.readline()
    f.close()
    line = line.strip()
    return line

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

    cutt_path = _find_cutt__()
    if not cutt_path is None:
        out += " " + os.path.join(cutt_path,"lib/libcutt.a")


    f.close()
    return out

def _get_variant_info__():
    f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"vinfo.tmp"),'r')
    lines = f.readlines()
    out = []
    for line in lines:
        line = line.strip()
        out+=line.split()
    f.close()
    return "\n".join(out)

__cpp_linkflags__ = _resolve_cpp_linkflags__()
__cpp_flags__ = _resolve_cpp_compileflags__()
__version__ = _get_version__()
__variant_info__ = _get_variant_info__()
