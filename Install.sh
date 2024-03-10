#=========================================================
## 1) Custom install destination (DEFAULT /usr/local/cytnx)
#=========================================================
# [Note] Set the destination path for installation in Ins_dest
#----------------------------------------------
Ins_dest="/usr/local/cytnx"
FLAG="${FLAG} -DCMAKE_INSTALL_PREFIX=${Ins_dest}"
#-----------------------------------------------

#=================================================================
# 2) linalg libs:
#=================================================================
# [Note] we have two choice: OpenBLAS or MKL (choose either one!)
#        uncomments either a) or b) for the one you wish to use.
#---------------------------
# 2-a) use MKL (DEFAULT = OFF):
# [Note] if using mkl, uncomments the following line and
#        Please follow the guide from official mkl/oneMKL "post-installation" part
#        to source the proper setvars.sh and/or vars.sh
#---------------------------
FLAG="${FLAG} -DUSE_MKL=OFF"
#---------------------------
# 2-b) use OpenBLAS (DEFAULT = on (by settung DUSE_MKL=OFF above))
# [Note] By default it will automatically find openblas installed
#        In case of cmake cannot find the BLAS, uncomments the following lines
#        and set the BLAS_ROOT:
#---------------------------
#BLASROOT=/deps/OpenBLAS
#export C_INCLUDE_PATH=${BLASROOT}/include
#export CPLUS_INCLUDE_PATH=${BLASROOT}/include
#export LD_LIBRARY_PATH=${BLASROOT}/lib
#---------------------------


#============================================================================
# 3) use python (DEFAULT = ON)
#============================================================================
# set to "=on" for building python API, required python and pybind11 installed.
#--------------------------------
FLAG="${FLAG} -DBUILD_PYTHON=ON"
#--------------------------------


#=================================================================
# 4) torch backend (DEFAULT = OFF):
#=================================================================
# [Note] if set to ON, torch will be used as a backend container
#				 for tensors. This allows to use automatic differentiatio
#--------------------------------
FLAG="${FLAG} -DBACKEND_TORCH=OFF"
#--------------------------------


#=========================================================
# 5) Boost:
#=========================================================
# [Note] by default, it will automatically find system installed boost.
#        In case boost cannot be found automatically, uncomment the following line,
#        and set the boost path manually.
#-----------------------------------
#BOOST_ROOT=/deps/BOOST
#FLAG="${FLAG} -DBOOSTROOT=${BOOST_ROOT}"
#-----------------------------------


#=========================================================
# 6) HPTT (DEFAULT = OFF):
#=========================================================
# [Note] set to "=on" for using hptt library to accelrate tensor transpose.
#        for "=off" case one can skip 5-a) and  5-b)
#-----------------------------------
FLAG="${FLAG} -DUSE_HPTT=OFF"
#-----------------------------------
# 6-a) HPTT fine tune (DEFAULT = OFF)
# [Note] set to "=on" to enable fine tune for the native hardware.
#-----------------------------------
FLAG="${FLAG} -DHPTT_ENABLE_FINE_TUNE=OFF"
#-----------------------------------
# 6-b) HPTT variant options (DEFAULT = no option)
# [Note] uncomment one of the desired options below 1: AVX 2: IBM 3: ARM.
#-----------------------------------
#FLAG="${FLAG} -DHPTT_ENABLE_ARM=ON"
#FLAG="${FLAG} -DHPTT_ENABLE_AVX=ON"
#FLAG="${FLAG} -DHPTT_ENABLE_IBM=ON"
#-----------------------------------


#=========================================================
# 7) CUDA (DEFAULT = OFF):
#=========================================================
# [Note] set to "=on" to build with with GPU (CUDA) support.
#        for "=off" case one can skip 6-a) and  6-b)
#-----------------------------------
FLAG="${FLAG} -DUSE_CUDA=OFF"
#-----------------------------------
# 7-a) CUTT (DEFAULT = OFF)
# [Note] set to "=on" for using CUTT library to accelrate tensor transpose.
#        for "=off" case one can skip 6-b)
#-----------------------------------
FLAG="${FLAG} -DUSE_CUTT=OFF"
#-----------------------------------
# 7-b) CUTT fine tune (DEFAULT = OFF)
# [Note] set to "=on" to enable fine tune for the native hardware.
#-----------------------------------
FLAG="${FLAG} -DCUTT_ENABLE_FINE_TUNE=OFF"
#-----------------------------------
# 7-c) Magma (DEFAULT = OFF)
# [Note] set to "=off" will make some of the GPU functions unavailable.
#        in case MAGMA is not automatically found, please specify MAGMAROOT path
#        where the magma is installed
# [Remark] You need to install MAGMA with BLA_VENDOR Intel10_64_dyn or Intel10_64ilp
#-----------------------------------
FLAG="${FLAG} -DUSE_MAGMA=OFF"
#CUTENSOR_ROOT=/usr/local/magma-...
#FLAG="${FLAG} -DMAGMA_ROOT=${MAGMA_ROOT}"
#-----------------------------------
# 7-d) CuTensor (DEFAULT = OFF)
# [Note] set to "=off" will make permutation on GPU into using cutt library.
# [Note] CUTENSOR_ROOT is required to given, either from enviroment variable in bashrc
#        or given in the following line using -DCUTENSOR_ROOT
FLAG="${FLAG} -DUSE_CUTENSOR=OFF"
#CUTENSOR_ROOT=/usr/local/libcutensor-...
#FLAG="${FLAG} -DCUTENSOR_ROOT=${CUTENSOR_ROOT}"
#-----------------------------------
# 7-e) CuQuantum (DEFAULT = OFF)
# [Note] CUQUANTUM_ROOT is required to given, either from enviroment variable in bashrc
#        or given in the following line using -DCUTENSOR_ROOT
FLAG="${FLAG} -DUSE_CUQUANTUM=OFF"
# CUQUANTUM_ROOT=/usr/local/cuqunatum-......
#FLAG="${FLAG} -DCUQUANTUM_ROOT=${CUQUANTUM_ROOT}"
#-----------------------------------


#=========================================================
# 8) Generate compile_commands.json
#=========================================================
# [Note] Wheather to generate compile_commands.json for IDE support (DEFAULT = 1)
#-----------------------------------
FLAG="${FLAG} -DCMAKE_EXPORT_COMPILE_COMMANDS=1"
#-----------------------------------


#=========================================================
# 9) Use openmp to accelerate
#=========================================================
# [Note] This will run the threaded code in cytnx lib while it will not disable muiltithreading in mkl. (DEFAULT = OFF)
#-----------------------------------
FLAG="${FLAG} -DUSE_OMP=OFF"
#-----------------------------------


#=========================================================
# 10) Run tests
#=========================================================
# [Note] Wheather to run cytnx tests (DEFAULT = OFF)
#-----------------------------------
DRUN_TESTS=OFF
FLAG="${FLAG} -DRUN_TESTS=${DRUN_TESTS}"
#-----------------------------------


#=========================================================
# 11) Use icpc
#=========================================================
# [Note] Build using intel icpc compiler, uncomment to enable (DEFAULT = OFF (commented out))
#-----------------------------------
#FLAG="${FLAG} -DUSE_ICPC=ON " # You should set this to ON if you want to use icpc
# This option is needed if you want to use icpc, to set the compiler
# Although icc&icpc is deprecated, but since icx&icpx has poor support for conda environment, we still use icpc
#FLAG="${FLAG} -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc"
# This option is to use the intel compiler's ar and linker, which is optional
#FLAG = "${FLAG} -DCMAKE_AR=xiar -DCMAKE_LINKER=xild"
#-----------------------------------


#=========================================================
# 12) Use Debug
#=========================================================
# [Note] Build using debug mode (DEFAULT = OFF)
#-----------------------------------
FLAG="${FLAG} -DUSE_DEBUG=OFF"
# This is for compile with -fsanitize=address and cuda,
# if you use the DEBUG flag above, you need to export ASAN_OPTIONS=protect_shadow_gap=0
# export ASAN_OPTIONS=protect_shadow_gap=0
# Just a note: export ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0
#-----------------------------------


#=========================================================
# Build commands
#=========================================================
echo ${FLAG}
rm -rf build
mkdir build
cd build
cmake ../ ${FLAG} #-DDEV_MODE=on
make -j`nproc`
make install
# if DRUN_TESTS=ON, run tests
# ctest
# shopt -s nocasematch
# case "${DRUN_TESTS}" in
#  "ON" ) ctest; gcovr -r ../ . --html-details cov.html;;
#  *) echo "Tests are skipped";;
# esac
