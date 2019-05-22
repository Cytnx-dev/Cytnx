import platform
from distutils.core import setup, Extension
from distutils import sysconfig

tor10_api = ["../uni10/src/Storage.cpp", 
			 "../uni10/src/Uint64Storage.cpp", 
			 "../uni10/src/Int64Storage.cpp", 
			 "../uni10/src/Uint32Storage.cpp", 
			 "../uni10/src/Int32Storage.cpp", 
			 "../uni10/src/DoubleStorage.cpp", 
			 "../uni10/src/FloatStorage.cpp", 
			 "../uni10/src/ComplexDoubleStorage.cpp", 
			 "../uni10/src/ComplexFloatStorage.cpp", 
             "../uni10/src/ComplexFloatStorage.cpp",
			 "../uni10/src/uni10_api/network_tools/pseudotensor.cpp"]

lapack_dir = ["../uni10/src/uni10_lapack_cpu/uni10_elem_lapack_cpu.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_add.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_add_v.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_conj.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_dagger.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_det.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_dot.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_eig.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_eigh.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_exp_v.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_identity.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_inverse.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_ldq.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_lq.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_mul.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_mul_v.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_norm.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_normalrandomize.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_qdr.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_qdrcpivot.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_ql.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_qr.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_r2c.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_rq.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_scal_v.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_sdd.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_setdiag.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_sub.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_sub_v.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_sum_v.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_svd.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_sytrimateigdcp.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_trace.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_transpose.cpp", 
			  "../uni10/src/uni10_lapack_cpu/linalg/uni10_elem_uniformrandomize.cpp", 
			  "../uni10/src/uni10_lapack_cpu/hirnk_linalg/hptt_tcl/uni10_elem_tensorcontract.cpp", 
			  "../uni10/src/uni10_lapack_cpu/hirnk_linalg/hptt_tcl/uni10_elem_tensortranspose.cpp", 
			  "../uni10/src/uni10_lapack_cpu/hirnk_linalg/ttgt/uni10_elem_tensorcontract.cpp", 
			  "../uni10/src/uni10_lapack_cpu/hirnk_linalg/ttgt/uni10_elem_tensortranspose.cpp", 
			  "../uni10/src/uni10_lapack_cpu/tools_lapack_cpu/uni10_tools_lapack_cpu.cpp", 
			  "../uni10/src/uni10_lapack_cpu/tools_lapack_cpu/uni10_linalg_lapack_cpu_d.cpp", 
			  "../uni10/src/uni10_lapack_cpu/tools_lapack_cpu/uni10_linalg_lapack_cpu_dz.cpp", 
			  "../uni10/src/uni10_lapack_cpu/tools_lapack_cpu/uni10_linalg_lapack_cpu_z.cpp"]

env_info = ["../uni10/src/uni10_env_info/uni10_lapack_cpu/uni10_env_info_lapack_cpu.cpp"]

cpp_args = ["-std=gnu++0x", "-DUNI_CPU", "-DUNI_LAPACK", "-D_hypot=hypot", "-Wno-sign-compare", "-Wno-unused-function", "-Wno-unused-variable"]

wrap = ["pytor10.cpp"]

src = uni10_api + lapack_dir + env_info + wrap

if platform.architecture()[0] == "32bit":
	lib_dir = "dep/x86"
else:
	lib_dir = "dep/x64"

ext_modules = [
	Extension("pyUni10", sources=src, 
			  include_dirs=["pybind11/include", "../uni10/include"], 
			  library_dirs=[lib_dir],
			  libraries=["blas", "lapack"],
			  language="c++", extra_compile_args=cpp_args)]

setup(
	name="pyUni10", version="2.0.0", ext_modules=ext_modules)

