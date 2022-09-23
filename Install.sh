
#=========================================================
## 1) Custom install destination (DEFAULT /use/local/cytnx)
#=========================================================
## [Note] uncomments the following lines and 
##        Set the desntination path for installation in Ins_dest
#----------------------------------------------
Ins_dest="/home/kaihsinwu/CYTNX077"
FLAG="${FLAG}-DCMAKE_INSTALL_PREFIX=${Ins_dest} "
#-----------------------------------------------


#=================================================================
# 2) linalg libs:
# [Note] we have two choice: OpenBLAS or MKL (choose either one!)
#        uncomments either a) or b) for the one you wish to use. 
#=================================================================

# 2-a) use OpenBLAS (DEFAULT)
# [Note] By default it will automatically find openblas installed
#        In case of cmake cannot find the BLAS, uncomments the following lines 
#        and set the BLAS_ROOT:
#---------------------------
#BLASROOT=/deps/OpenBLAS
#export C_INCLUDE_PATH=${BLASROOT}/include
#export CPLUS_INCLUDE_PATH=${BLASROOT}/include
#export LD_LIBRARY_PATH=${BLASROOT}/lib
#---------------------------

# 2-b) use MKL:
# [Note] if using mkl, uncomments the following line and
#        Please follow the guide from official mkl/oneMKL "post-installation" part 
#        to source the proper setvars.sh and/or vars.sh  
#----------------------
FLAG="${FLAG}-DUSE_MKL=ON "
#----------------------


#============================================================================
# 3) use python (DEFAULT =on)
#============================================================================
# set to "=on" for building python API, required python and pybind11 installed. 
#--------------------------------
FLAG="${FLAG}-DBUILD_PYTHON=ON "
#--------------------------------


#=========================================================
# 4) Boost:
#=========================================================
# [Note] by default, it will automatically find system installed boost.
#        In case boost cannot be found automatically, uncomment the following line,
#        and set the boost path manually.  
#-----------------------------------
#FLAG="${FLAG}-DBOOSTROOT=/deps/BOOST "
#-----------------------------------


#=========================================================
# 5) HPTT (DEFAULT =off):
#=========================================================
# [Note] set to "=on" for using hptt library to accelrate tensor transpose.
#        for "=off" case one can skip 5-a) and  5-b)
#-----------------------------------
FLAG="${FLAG}-DUSE_HPTT=OFF "
#-----------------------------------
# 5-a) HPTT fine tune (DEFAULT =off)
# [Note] set to "=on" to enable fine tune for the native hardware.
#-----------------------------------
# FLAG="${FLAG}-DHPTT_ENABLE_FINE_TUNE=off"
#-----------------------------------
# 5-b) HPTT variant options (DEFAULT = no option)
# [Note] uncomment one of the desired options below 1: AVX 2: IBM 3: ARM.
#-----------------------------------
# FLAG="${FLAG}-DHPTT_ENABLE_ARM=on"
# FLAG="${FLAG}-DHPTT_ENABLE_AVX=on"
# FLAG="${FLAG}-DHPTT_ENABLE_IBM=on"
#-----------------------------------


#=========================================================
# 6) CUDA (DEFAULT =off):
#=========================================================
# [Note] set to "=on" to build with with GPU (CUDA) support.
#        for "=off" case one can skip 6-a) and  6-b)
#-----------------------------------
FLAG="${FLAG}-DUSE_CUDA=OFF "
#-----------------------------------
# 6-a) CUTT (DEFAULT =off)
# [Note] set to "=on" for using CUTT library to accelrate tensor transpose.
#        for "=off" case one can skip 6-b)
#-----------------------------------
# FLAG="${FLAG}-DUSE_CUTT=off"
#-----------------------------------
# 6-b) CUTT fine tune (DEFAULT =off)
# [Note] set to "=on" to enable fine tune for the native hardware.
#-----------------------------------
# FLAG="${FLAG}-DCUTT_ENABLE_FINE_TUNE=off"
#-----------------------------------


#=========================================================
# 7) Generate compile_commands.json (DEFAULT =off):
#=========================================================
# [Note] Wheather to generate compile_commands.json for IDE support (DEFAULT =1)
#-----------------------------------
FLAG="${FLAG}-DCMAKE_EXPORT_COMPILE_COMMANDS=0 "
#-----------------------------------


#=========================================================
# 8) Use openmp to accelerate
#=========================================================
# [Note] This will run the threaded code in cytnx lib while it will not disable muiltithreading in mkl. (DEFAULT =off)
#-----------------------------------
FLAG="${FLAG}-DUSE_OMP=OFF "
#-----------------------------------


#=========================================================
# 9) Run tests
#=========================================================
# [Note] Wheather to run cytnx tests (DEFAULT =off)
#-----------------------------------
FLAG="${FLAG}-DRUN_TESTS=OFF "
#-----------------------------------


#=========================================================
# 9) Use icpc ?
#=========================================================
# [Note] Build using intel icpc compiler (DEFAULT =off)
#-----------------------------------
FLAG="${FLAG}-DUSE_ICPC=OFF "
#-----------------------------------


echo ${FLAG}
rm -rf build
mkdir build
cd build
cmake ../ ${FLAG}
make -j`nproc`
make install
