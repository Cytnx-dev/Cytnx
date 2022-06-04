
#=========================================================
## 1) Custom install destination (DEFAULT /use/local/cytnx)
#=========================================================
## [Note] uncomments the following lines and 
##        Set the desntination path for installation in Ins_dest
#----------------------------------------------
Ins_dest="/home/kaihsinwu/CYTNX075"
FLAG="${FLAG}-DCMAKE_INSTALL_PREFIX=${Ins_dest} "
#-----------------------------------------------


#=================================================================
## 2) linalg libs:
## [Note] we have two choice: OpenBLAS or MKL (choose either one!)
##        uncomments either a) or b) for the one you wish to use. 
#=================================================================

# a) use OpenBLAS (DEFAULT)
# [Note] By default it will automatically find openblas installed
#        In case of cmake cannot find the BLAS, uncomments the following lines 
#        and set the BLAS_ROOT:
#---------------------------
#BLASROOT=/deps/OpenBLAS
#export C_INCLUDE_PATH=${BLASROOT}/include
#export CPLUS_INCLUDE_PATH=${BLASROOT}/include
#export LD_LIBRARY_PATH=${BLASROOT}/lib
#---------------------------


# b) use MKL:
# [Note] if using mkl, uncomments the following line and
#        Please follow the guide from official mkl/oneMKL "post-installation" part 
#        to source the proper setvars.sh and/or vars.sh  
#----------------------
FLAG="${FLAG}-DUSE_MKL=on "
#----------------------


#============================================================================
# 3) use python (DEFAULT =on)
#============================================================================
# set to "=on" for building python API, required python and pybind11 installed. 
#--------------------------------
#FLAG=$FLAG:"-DBUILD_PYTHON=off "
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


# [Optional]
FLAG="${FLAG}-DUSE_HPTT=on "


rm -rf build
mkdir build
cd build
cmake ../ ${FLAG}
make -j6
