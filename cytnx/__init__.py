import os,sys,warnings
from . import _cuda_preload

if _cuda_preload.is_cuda_build(os.path.dirname(os.path.abspath(__file__))):
    _cuda_preload.preload()

from . import cytnx
from .cytnx import *
from .cytnx import __version__

if cytnx.__cytnx_backend__ == "torch":
    import numpy, torch

    from .Symmetry_conti import *
    from .Bond_conti import *

else:

    import numpy


    from .Storage_conti import *
    from .Tensor_conti import *
    from .linalg_conti import *
    from .UniTensor_conti import *
    from .Network_conti import *
    from .Symmetry_conti import *
    from .Bond_conti import *

    def from_numpy(np_arr):
        tmp = np_arr
        if np_arr.flags['C_CONTIGUOUS'] == False:
            tmp = numpy.ascontiguousarray(np_arr)
        return cytnx._from_numpy(tmp)


    def _validate_blas_runtime():
        # When cytnx is linked against a BLAS/LAPACK build with a different
        # integer width than another library loaded in the same process (e.g.
        # MKL ILP64 vs LP64), the dynamic loader resolves the shared symbols
        # to whichever side was loaded first; cytnx then calls them with the
        # wrong argument layout. Whether any particular routine actually
        # misfires depends on how many int args it takes and what ends up next
        # to them in the caller's frame: NumPy's own _sanity_check is a single
        # `float32` dot of size 2 and misses the failure modes reported in
        # #595/#419/#765 (ZHEEVD prints "Parameter N was incorrect", DGESDD
        # aborts in `init_gesdd`, and ZGEMM silently returns wrong values).
        # Probe each of the three call sites actually reported broken, plus
        # the SDOT-equivalent NumPy itself probes, and raise ImportError with
        # a clear diagnostic if any of them comes back wrong. Guarding numpy's
        # own calls is numpy's responsibility (via its existing _sanity_check);
        # we only assert cytnx's BLAS path is sane.
        def _fail(routine, observed, expected, exc=None):
            raise ImportError(
                "cytnx: BLAS/LAPACK runtime check failed in {}.\n"
                "  observed: {}\n"
                "  expected: {} (within 1e-6)\n\n"
                "This usually means cytnx and numpy/scipy are linked against "
                "BLAS builds with different integer widths (e.g. MKL ILP64 vs "
                "LP64), so the dynamic loader resolves shared LAPACK symbols "
                "to one side and cytnx calls them with the wrong argument "
                "layout. See "
                "https://github.com/Cytnx-dev/Cytnx/issues/595 for "
                "troubleshooting; in particular, compare the BLAS library "
                "numpy is linked against (numpy.show_config()) with the one "
                "in cytnx.__cpp_linkflags__ and make sure their integer "
                "widths match.".format(routine, observed,
                                       expected)) from exc

        def _probe(routine, expected, fn):
            try:
                observed = fn()
            except Exception as exc:
                _fail(routine, "<raised {!r}>".format(exc), expected, exc=exc)
            if len(observed) != len(expected) or not numpy.allclose(
                    observed, expected, atol=1e-6):
                _fail(routine, observed, expected)

        # --- SDOT probe (mirrors NumPy's _sanity_check on the cytnx side)
        def _sdot():
            x = cytnx.ones(2, dtype=cytnx.Type.Float)
            r = cytnx.linalg.Dot(x, x)
            return [float(numpy.asarray(r.numpy()).ravel()[0])]

        _probe("linalg.Dot (SDOT)", [2.0], _sdot)

        # --- ZGEMM probe (covers #765: complex Contract returning wrong value)
        def _zgemm():
            A = cytnx.zeros([2, 2], dtype=cytnx.Type.ComplexDouble)
            A[0, 0] = 1.0
            A[1, 1] = 1.0j
            B = cytnx.zeros([2, 2], dtype=cytnx.Type.ComplexDouble)
            B[0, 0] = 2.0
            B[0, 1] = 3.0
            B[1, 0] = 5.0
            B[1, 1] = 7.0 + 11.0j
            C = cytnx.linalg.Matmul(A, B)
            return [complex(x) for x in numpy.asarray(C.numpy()).reshape(-1)]

        _probe("linalg.Matmul (ZGEMM)", [2 + 0j, 3 + 0j, 0 + 5j, -11 + 7j],
               _zgemm)

        # --- ZHEEVD probe (covers #595: Hermitian eigendecomposition)
        def _zheevd():
            H = cytnx.zeros([2, 2], dtype=cytnx.Type.ComplexDouble)
            H[0, 0] = 2.0
            H[1, 1] = 5.0
            H[0, 1] = -3.0j
            H[1, 0] = 3.0j
            eigvals = cytnx.linalg.Eigh(H, is_V=False)[0]
            return sorted(float(x) for x in
                          numpy.asarray(eigvals.numpy()).real.ravel())

        _probe("linalg.Eigh (ZHEEVD)",
               sorted([(7.0 - 45.0 ** 0.5) / 2.0,
                       (7.0 + 45.0 ** 0.5) / 2.0]),
               _zheevd)

        # --- DGESDD probe (covers #419: real SVD)
        def _dgesdd():
            M = cytnx.zeros([2, 2], dtype=cytnx.Type.Double)
            M[0, 0] = 1.0
            M[0, 1] = 2.0
            M[1, 0] = 3.0
            M[1, 1] = 4.0
            sv = cytnx.linalg.Svd(M, is_UvT=False)[0]
            return sorted((float(x) for x in
                           numpy.asarray(sv.numpy()).ravel()), reverse=True)

        _probe("linalg.Svd (DGESDD)",
               [5.464985704219043, 0.3659661906262571], _dgesdd)


    if os.environ.get("CYTNX_SKIP_BLAS_RUNTIME_CHECK", "") not in ("", "0"):
        warnings.warn(
            "Skipping cytnx BLAS/LAPACK runtime check "
            "(CYTNX_SKIP_BLAS_RUNTIME_CHECK is set). Misaligned BLAS "
            "interfaces between cytnx and numpy/scipy may silently corrupt "
            "results; see https://github.com/Cytnx-dev/Cytnx/issues/595.")
    else:
        _validate_blas_runtime()



__blasINTsize__ = cytnx.__blasINTsize__



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
__variant_info__ = _get_variant_info__()
