CONFIG_FILE := make.inc

##########################################
# Color
COLOR_REST=\e[0m
COLOR_GREEN=\e[0;32m
COLOR_RED=\e[0;31m
COLOR_PURPLE=\e[0;35m
COLOR_YELLOW=\e[0;33m

## Explicitly check for the config file, otherwise make -k will proceed anyway.
ifeq ($(wildcard $(CONFIG_FILE)),)
$(error $(CONFIG_FILE) not found.)
endif
include $(CONFIG_FILE)

## Detecting the OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  PYOBJFLAGS := -undefined dynamic_lookup
else
  PYOBJFLAGS := 
endif


## 
CytnxPATH=.
INCFLAGS :=-I$(CytnxPATH)/include -I$(CytnxPATH)/src

HPTT_PATH=./thirdparty/hptt
CUTT_PATH=./thirdparty/cutt


ifeq ($(ICPC_Enable),1)
  CC:= $(ICPC)
  CCFLAGS:= -DUNI_ICPC
else
  CC:= $(GCC)
  CCFLAGS:= 
endif

ifeq ($(MKL_Enable),1)
  CCFLAGS += -std=c++11 ${OPTIM} -Wformat=0 -m64 -fPIC -DUNI_MKL -w -DMKL_ILP64 -Wno-c++11-narrowing #-DUNI_DEBUG -Wno-c++11-narrowing
  LDFLAGS += $(DOCKER_MKL) -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl -lm 
else
  CCFLAGS += -std=c++11 ${OPTIM} -Wformat=0 -fPIC -w -g #-DUNI_DEBUG -Wno-c++11-narrowing
  LDFLAGS += -llapacke -lblas -lstdc++  
endif


NVCC:= $(CUDA_PATH)/bin/nvcc -ccbin $(CC)
NVFLAGS:= -m64 ${OPTIM} -lineinfo
SMS ?= 60
GENCODE_FLAGS:= -arch=sm_$(SMS)

ifeq ($(OMP_Enable),1)
	ifeq ($(MKL_Enable),1)
		CCFLAGS += -DUNI_OMP -fopenmp
	else
		CCFLAGS += -DUNI_OMP -fopenmp
	endif
endif

ifeq ($(DEBUG_Enable),1)
  CCFLAGS += -DUNI_DEBUG
endif

ifeq ($(HPTT_Enable),1)
  CCFLAGS += -DUNI_HPTT 
  INCFLAGS+= -I$(HPTT_PATH)/include
  LDFLAGS += $(HPTT_PATH)/lib/libhptt.a
endif


ALL_CCFLAGS := 
ifeq ($(GPU_Enable),1)
  ifeq ($(CUTT_Enable),1)
   CCFLAGS += -DUNI_CUTT
   INCFLAGS+= -I$(CUTT_PATH)/include
   LDFLAGS+= $(CUTT_PATH)/lib/libcutt.a
  endif

  CCFLAGS += -I$(CUDA_PATH)/include -DUNI_GPU
  ALL_CCFLAGS += $(NVFLAGS)
  ALL_CCFLAGS += $(addprefix -Xcompiler , $(CCFLAGS))
  ALL_CCFLAGS += $(addprefix -Xcompiler , $(INCFLAGS))
  ALL_CCFLAGS += $(GENCODE_FLAGS)
else
  ALL_CCFLAGS += $(CCFLAGS)
endif

ALL_LDFLAGS :=
ifeq ($(GPU_Enable),1)
  LDFLAGS += -lcublas -lcusolver -lcurand -lcudart
  ALL_LDFLAGS += $(addprefix -Xlinker , $(LDFLAGS)) 
  ALL_LDFLAGS += -L$(CUDA_PATH)/lib64
  LDFLAGS += -L$(CUDA_PATH)/lib64 
else
  ALL_LDFLAGS += $(LDFLAGS)
endif

OBJS = Scalar.o Storage_base.o BoolStorage.o Uint16Storage.o Int16Storage.o Uint32Storage.o Int32Storage.o Uint64Storage.o Int64Storage.o FloatStorage.o DoubleStorage.o ComplexFloatStorage.o ComplexDoubleStorage.o Type.o Device.o


OBJS += LinOp.o Storage.o Tensor.o Accessor.o Generator.o Physics.o
OBJS += Network.o Network_base.o RegularNetwork.o FermionNetwork.o UniTensor_base.o DenseUniTensor.o SparseUniTensor.o UniTensor.o Bond.o Symmetry.o contraction_tree.o search_tree.o

## TN 
OBJS += DMRG.o MPO.o RegularMPO.o MPO_base.o MPS.o RegularMPS.o iMPS.o MPS_base.o



## Utils
OBJS += utils_internal_interface.o
OBJS += utils.o Cast_cpu.o Alloc_cpu.o Movemem_cpu.o Range_cpu.o vec_io.o vec_print.o vec2d_col_sort.o vec_range.o complex_arithmetic.o is.o vec_intersect.o vec_concatenate.o vec_where.o vec_erase.o vec_clone.o vec_unique.o vec_map.o SetZeros_cpu.o Fill_cpu.o SetArange_cpu.o GetElems_contiguous_cpu.o GetElems_cpu.o SetElems_contiguous_cpu.o SetElems_cpu.o cartesian.o str_utils.o Complexmem_cpu.o dynamic_arg_resolver.o blocks_mvelems_cpu.o 
ifeq ($(GPU_Enable),1)
  OBJS += cucomplex_arithmetic.o cuAlloc_gpu.o cuCast_gpu.o cuMovemem_gpu.o cuSetZeros_gpu.o cuFill_gpu.o cuSetArange_gpu.o cuGetElems_gpu.o  cuSetElems_gpu.o cuComplexmem_gpu.o cuReduce_gpu.o
endif

## Linalg_internal
OBJS += linalg_internal_interface.o
OBJS += Lstsq_internal.o Mod_internal.o Det_internal.o Sum_internal.o MaxMin_internal.o QR_internal.o Abs_internal.o Pow_internal.o Eig_internal.o Matvec_internal.o Norm_internal.o Kron_internal.o Cpr_internal.o iAdd_internal.o Add_internal.o iSub_internal.o Sub_internal.o iMul_internal.o Mul_internal.o iDiv_internal.o Div_internal.o iArithmetic_internal.o Arithmetic_internal.o Svd_internal.o Inv_inplace_internal.o InvM_inplace_internal.o Conj_inplace_internal.o Exp_internal.o Eigh_internal.o Matmul_dg_internal.o Matmul_internal.o Diag_internal.o Outer_internal.o Vectordot_internal.o Tridiag_internal.o 
ifeq ($(GPU_Enable),1)
  OBJS += cuMod_internal.o cuPow_internal.o cuVectordot_internal.o cuMatvec_internal.o cuNorm_internal.o cuCpr_internal.o cuAdd_internal.o cuSub_internal.o cuMul_internal.o cuDiv_internal.o cuArithmetic_internal.o cuSvd_internal.o cuInv_inplace_internal.o cuInvM_inplace_internal.o cuConj_inplace_internal.o cuExp_internal.o  cuEigh_internal.o cuMatmul_dg_internal.o cuMatmul_internal.o cuDiag_internal.o cuOuter_internal.o
endif

## Algo_internal
OBJS += algo_internal_interface.o
OBJS += Sort_internal.o



## Linalg
OBJS += Lstsq.o Mod.o Lanczos_Gnd_Ut.o Lanczos_Gnd.o Lanczos_ER.o Det.o Sum.o Hosvd.o Min.o Max.o ExpM.o Qdr.o Qr.o Abs_.o Abs.o Pow_.o Pow.o Trace.o Eig.o Dot.o Norm.o ExpH.o Kron.o iAdd.o Add.o iDiv.o Div.o iSub.o Sub.o iMul.o Mul.o Cpr.o Svd.o Svd_truncate.o Inv.o Inv_.o InvM.o InvM_.o Conj.o Conj_.o Exp.o Exp_.o Expf.o Expf_.o Eigh.o Diag.o Matmul_dg.o Matmul.o Tensordot_dg.o Tensordot.o Outer.o Vectordot.o Tridiag.o 

## Algo
OBJS += Sort.o Concatenate.o  

## Stat
OBJS += histogram.o 

## Random_internal
OBJS += random_internal_interface.o
OBJS += Normal_internal.o Uniform_internal.o
ifeq ($(GPU_Enable),1)
  OBJS += cuNormal_internal.o cuUniform_internal.o
endif

## Random
OBJS += Make_normal.o normal.o Make_uniform.o uniform.o 


ALLOBJS = $(OBJS)
##
ifeq ($(GPU_Enable),1)
  ALLOBJS += gpusrc.o
endif

TESTPATH=tests


all: test dmrg_tfim


#test: test.o $(ALLOBJS)
#	#$(CC) -o $@ $^ $(CCFLAGS) $(LDFLAGS)
#	$(NVCC) $(ALL_CCFLAGS) $(ALL_LDFLAGS) $^ -o $@

test: test.o libcytnx.so
	$(CC) -L. -o $@ $< -fopenmp $(LDFLAGS) -lcytnx 
	#export LD_LIBRARY_PATH=.

dmrg_tfim: dmrg_tfim.o libcytnx.so
	$(CC) -L. -o $@ $< -fopenmp $(LDFLAGS) -lcytnx 
	#export LD_LIBRARY_PATH=.

#dmrg_tfim2: dmrg_tfim2.o libcytnx.so
#	$(CC) -L. -o $@ $< -fopenmp $(LDFLAGS) -lcytnx 
#	#export LD_LIBRARY_PATH=.

libcytnx.so: $(ALLOBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

pyobj: $(ALLOBJS)
	$(CC) $(INCFLAGS) $(CCFLAGS) $(PYOBJFLAGS) $(shell python3 -m pybind11 --includes)  pybind/cytnx.cpp $^ $(LDFLAGS) -shared -o cytnx/cytnx$(shell python3-config --extension-suffix)


#cytnx.o: pybind/cytnx.cpp
#	$(CC) -c $(INCFLAGS) $(CCFLAGS) $(PYOBJFLAGS) $(shell python3 -m pybind11 --includes)  $< -o $@

#pyobj: cytnx.o 
#	$(CC) -L. $< -shared -o cytnx/cytnx$(shell python3-config --extension-suffix) $(LDFLAGS) -lcytnx



doc : 
	doxygen docs.doxygen


ifeq ($(GPU_Enable),1)
gpusrc.o: $(OBJS)
	$(NVCC) $(ALL_CCFLAGS) $(ALL_LDFLAGS) -dlink $^ -o $@ 
endif

## Tensor
Tensor.o: $(CytnxPATH)/src/Tensor.cpp $(CytnxPATH)/include/Tensor.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Storage.o: $(CytnxPATH)/src/Storage.cpp $(CytnxPATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Accessor.o: $(CytnxPATH)/src/Accessor.cpp $(CytnxPATH)/include/Accessor.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Generator.o: $(CytnxPATH)/src/Generator.cpp $(CytnxPATH)/include/Generator.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Physics.o: $(CytnxPATH)/src/Physics.cpp $(CytnxPATH)/include/Physics.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Bond.o : $(CytnxPATH)/src/Bond.cpp $(CytnxPATH)/include/Bond.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
Symmetry.o: $(CytnxPATH)/src/Symmetry.cpp $(CytnxPATH)/include/Symmetry.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
UniTensor.o: $(CytnxPATH)/src/UniTensor.cpp $(CytnxPATH)/include/UniTensor.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

UniTensor_base.o: $(CytnxPATH)/src/UniTensor_base.cpp $(CytnxPATH)/include/UniTensor.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
DenseUniTensor.o: $(CytnxPATH)/src/DenseUniTensor.cpp $(CytnxPATH)/include/UniTensor.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
SparseUniTensor.o: $(CytnxPATH)/src/SparseUniTensor.cpp $(CytnxPATH)/include/UniTensor.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Network.o: $(CytnxPATH)/src/Network.cpp $(CytnxPATH)/include/Network.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Network_base.o: $(CytnxPATH)/src/Network_base.cpp $(CytnxPATH)/include/Network.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
RegularNetwork.o: $(CytnxPATH)/src/RegularNetwork.cpp $(CytnxPATH)/include/Network.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
FermionNetwork.o: $(CytnxPATH)/src/FermionNetwork.cpp $(CytnxPATH)/include/Network.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
contraction_tree.o: $(CytnxPATH)/src/contraction_tree.cpp $(CytnxPATH)/include/contraction_tree.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
search_tree.o: $(CytnxPATH)/src/search_tree.cpp $(CytnxPATH)/include/search_tree.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
LinOp.o: $(CytnxPATH)/src/LinOp.cpp $(CytnxPATH)/include/LinOp.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

##--------------------------------------------------------------------------


## Storage 
###########################
Storage_base.o: $(CytnxPATH)/src/Storage_base.cpp $(CytnxPATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
BoolStorage.o: $(CytnxPATH)/src/BoolStorage.cpp $(CytnxPATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Uint16Storage.o: $(CytnxPATH)/src/Uint16Storage.cpp $(CytnxPATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Int16Storage.o: $(CytnxPATH)/src/Int16Storage.cpp $(CytnxPATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Uint32Storage.o: $(CytnxPATH)/src/Uint32Storage.cpp $(CytnxPATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Int32Storage.o: $(CytnxPATH)/src/Int32Storage.cpp $(CytnxPATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Uint64Storage.o: $(CytnxPATH)/src/Uint64Storage.cpp $(CytnxPATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Int64Storage.o: $(CytnxPATH)/src/Int64Storage.cpp $(CytnxPATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
FloatStorage.o: $(CytnxPATH)/src/FloatStorage.cpp $(CytnxPATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

DoubleStorage.o: $(CytnxPATH)/src/DoubleStorage.cpp $(CytnxPATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

ComplexFloatStorage.o: $(CytnxPATH)/src/ComplexFloatStorage.cpp $(CytnxPATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

ComplexDoubleStorage.o: $(CytnxPATH)/src/ComplexDoubleStorage.cpp $(CytnxPATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  


## algo_internal
##########################
algo_internal_interface.o : $(CytnxPATH)/src/algo/algo_internal_interface.cpp $(CytnxPATH)/src/algo/algo_internal_interface.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Sort_internal.o :  $(CytnxPATH)/src/algo/algo_internal_cpu/Sort_internal.cpp $(CytnxPATH)/src/algo/algo_internal_cpu/Sort_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

## algo
##########################
Sort.o : $(CytnxPATH)/src/algo/Sort.cpp $(CytnxPATH)/include/algo.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
Concatenate.o : $(CytnxPATH)/src/algo/Concatenate.cpp $(CytnxPATH)/include/algo.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

## stat
#############################
histogram.o: $(CytnxPATH)/src/stat/histogram.cpp $(CytnxPATH)/include/stat.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<


## TN
#############################
MPS.o: $(CytnxPATH)/src/tn_algo/MPS.cpp $(CytnxPATH)/include/tn_algo/MPS.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
RegularMPS.o: $(CytnxPATH)/src/tn_algo/RegularMPS.cpp $(CytnxPATH)/include/tn_algo/MPS.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
iMPS.o: $(CytnxPATH)/src/tn_algo/iMPS.cpp $(CytnxPATH)/include/tn_algo/MPS.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
MPS_base.o: $(CytnxPATH)/src/tn_algo/MPS_base.cpp $(CytnxPATH)/include/tn_algo/MPS.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
MPO.o: $(CytnxPATH)/src/tn_algo/MPO.cpp $(CytnxPATH)/include/tn_algo/MPO.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
RegularMPO.o: $(CytnxPATH)/src/tn_algo/RegularMPO.cpp $(CytnxPATH)/include/tn_algo/MPO.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
MPO_base.o: $(CytnxPATH)/src/tn_algo/MPO_base.cpp $(CytnxPATH)/include/tn_algo/MPO.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

DMRG.o: $(CytnxPATH)/src/tn_algo/DMRG.cpp $(CytnxPATH)/include/tn_algo/DMRG.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

## linalg_internal
###########################
linalg_internal_interface.o : $(CytnxPATH)/src/linalg/linalg_internal_interface.cpp $(CytnxPATH)/src/linalg/linalg_internal_interface.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Arithmetic_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Arithmetic_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Arithmetic_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
iArithmetic_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/iArithmetic_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/iArithmetic_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Cpr_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Cpr_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Cpr_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Add_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Add_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Add_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
iAdd_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/iAdd_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/iAdd_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Mul_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Mul_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Mul_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
iMul_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/iMul_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/iMul_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Sub_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Sub_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Sub_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
iSub_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/iSub_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/iSub_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Mod_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Mod_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Mod_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Div_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Div_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Div_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

iDiv_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/iDiv_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/iDiv_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Svd_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Svd_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Svd_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Eigh_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Eigh_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Eigh_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Exp_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Exp_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Exp_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Pow_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Pow_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Pow_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Diag_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Diag_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Diag_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

InvM_inplace_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/InvM_inplace_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/InvM_inplace_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Inv_inplace_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Inv_inplace_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Inv_inplace_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Conj_inplace_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Conj_inplace_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Conj_inplace_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Matmul_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Matmul_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Matmul_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Matmul_dg_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Matmul_dg_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Matmul_dg_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Matvec_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Matvec_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Matvec_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Outer_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Outer_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Outer_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Kron_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Kron_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Kron_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Vectordot_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Vectordot_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Vectordot_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Tridiag_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Tridiag_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Tridiag_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Norm_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Norm_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Norm_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Eig_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Eig_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Eig_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Abs_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Abs_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Abs_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
QR_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/QR_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/QR_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

MaxMin_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/MaxMin_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/MaxMin_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Sum_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Sum_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Sum_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Det_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Det_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Det_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Lstsq_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Lstsq_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Lstsq_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  


ifeq ($(GPU_Enable),1)

cuArithmetic_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuArithmetic_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuArithmetic_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuCpr_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuCpr_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuCpr_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuAdd_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuAdd_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuAdd_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuMul_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuMul_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuMul_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuSub_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuSub_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuSub_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuDiv_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuDiv_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuDiv_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuMod_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuMod_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuMod_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuSvd_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuSvd_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuSvd_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuEigh_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuEigh_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuEigh_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuExp_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuExp_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuExp_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuPow_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuPow_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuPow_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuDiag_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuDiag_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuDiag_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuInvM_inplace_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuInvM_inplace_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuInvM_inplace_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuInv_inplace_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuInv_inplace_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuInv_inplace_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuConj_inplace_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuConj_inplace_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuConj_inplace_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuMatmul_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuMatmul_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuMatmul_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuMatmul_dg_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuMatmul_dg_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuMatmul_dg_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuMatvec_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuMatvec_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuMatvec_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuVectordot_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuVectordot_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuVectordot_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuOuter_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuOuter_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuOuter_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuNorm_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuNorm_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuNorm_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

endif

## random_internal
###########################
random_internal_interface.o : $(CytnxPATH)/src/random/random_internal_interface.cpp $(CytnxPATH)/src/random/random_internal_interface.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Normal_internal.o :  $(CytnxPATH)/src/random/random_internal_cpu/Normal_internal.cpp $(CytnxPATH)/src/random/random_internal_cpu/Normal_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Uniform_internal.o :  $(CytnxPATH)/src/random/random_internal_cpu/Uniform_internal.cpp $(CytnxPATH)/src/random/random_internal_cpu/Uniform_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

ifeq ($(GPU_Enable),1)

cuNormal_internal.o :  $(CytnxPATH)/src/random/random_internal_gpu/cuNormal_internal.cu $(CytnxPATH)/src/random/random_internal_gpu/cuNormal_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuUniform_internal.o :  $(CytnxPATH)/src/random/random_internal_gpu/cuUniform_internal.cu $(CytnxPATH)/src/random/random_internal_gpu/cuUniform_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

endif
## Random:
########################
Make_normal.o: $(CytnxPATH)/src/random/Make_normal.cpp $(CytnxPATH)/include/random.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Make_uniform.o: $(CytnxPATH)/src/random/Make_uniform.cpp $(CytnxPATH)/include/random.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
normal.o: $(CytnxPATH)/src/random/normal.cpp $(CytnxPATH)/include/random.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
uniform.o: $(CytnxPATH)/src/random/uniform.cpp $(CytnxPATH)/include/random.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

## Misc
##########################
Type.o : $(CytnxPATH)/src/Type.cpp $(CytnxPATH)/include/Type.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Device.o: $(CytnxPATH)/src/Device.cpp $(CytnxPATH)/include/Device.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Scalar.o : $(CytnxPATH)/src/Scalar.cpp $(CytnxPATH)/include/Scalar.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

## Utils
##########################
utils_internal_interface.o: $(CytnxPATH)/src/utils/utils_internal_interface.cpp $(CytnxPATH)/src/utils/utils_internal_interface.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

utils.o: $(CytnxPATH)/src/utils/utils.cpp $(CytnxPATH)/include/utils/utils.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

Cast_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/Cast_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/Cast_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

Range_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/Range_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/Range_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

Movemem_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/Movemem_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/Movemem_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

Complexmem_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/Complexmem_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/Complexmem_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

Alloc_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/Alloc_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/Alloc_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

SetZeros_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/SetZeros_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/SetZeros_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

SetArange_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/SetArange_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/SetArange_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

GetElems_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/GetElems_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/GetElems_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
GetElems_contiguous_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/GetElems_contiguous_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/GetElems_contiguous_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

SetElems_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/SetElems_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/SetElems_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
SetElems_contiguous_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/SetElems_contiguous_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/SetElems_contiguous_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Fill_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/Fill_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/Fill_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

blocks_mvelems_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/blocks_mvelems_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/blocks_mvelems_cpu.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<


complex_arithmetic.o: $(CytnxPATH)/src/utils/complex_arithmetic.cpp $(CytnxPATH)/src/utils/complex_arithmetic.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

is.o: $(CytnxPATH)/src/utils/is.cpp $(CytnxPATH)/include/utils/is.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

cartesian.o: $(CytnxPATH)/src/utils/cartesian.cpp $(CytnxPATH)/include/utils/cartesian.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
vec_clone.o: $(CytnxPATH)/src/utils/vec_clone.cpp $(CytnxPATH)/include/utils/vec_clone.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
vec_unique.o: $(CytnxPATH)/src/utils/vec_unique.cpp $(CytnxPATH)/include/utils/vec_unique.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
vec_map.o: $(CytnxPATH)/src/utils/vec_map.cpp $(CytnxPATH)/include/utils/vec_map.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
vec_erase.o: $(CytnxPATH)/src/utils/vec_erase.cpp $(CytnxPATH)/include/utils/vec_erase.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
vec_where.o: $(CytnxPATH)/src/utils/vec_where.cpp $(CytnxPATH)/include/utils/vec_where.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
vec_range.o: $(CytnxPATH)/src/utils/vec_range.cpp $(CytnxPATH)/include/utils/vec_range.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
vec_print.o: $(CytnxPATH)/src/utils/vec_print.cpp $(CytnxPATH)/include/utils/vec_print.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
vec_io.o: $(CytnxPATH)/src/utils/vec_io.cpp $(CytnxPATH)/include/utils/vec_io.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

vec_concatenate.o: $(CytnxPATH)/src/utils/vec_concatenate.cpp $(CytnxPATH)/include/utils/vec_concatenate.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
vec_intersect.o: $(CytnxPATH)/src/utils/vec_intersect.cpp $(CytnxPATH)/include/utils/vec_intersect.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
vec2d_col_sort.o: $(CytnxPATH)/src/utils/vec2d_col_sort.cpp $(CytnxPATH)/include/utils/vec2d_col_sort.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
str_utils.o: $(CytnxPATH)/src/utils/str_utils.cpp $(CytnxPATH)/include/utils/str_utils.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
dynamic_arg_resolver.o: $(CytnxPATH)/src/utils/dynamic_arg_resolver.cpp $(CytnxPATH)/include/utils/dynamic_arg_resolver.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

ifeq ($(GPU_Enable),1)
cucomplex_arithmetic.o: $(CytnxPATH)/src/utils/cucomplex_arithmetic.cu $(CytnxPATH)/src/utils/cucomplex_arithmetic.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuAlloc_gpu.o: $(CytnxPATH)/src/utils/utils_internal_gpu/cuAlloc_gpu.cu $(CytnxPATH)/src/utils/utils_internal_gpu/cuAlloc_gpu.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuCast_gpu.o: $(CytnxPATH)/src/utils/utils_internal_gpu/cuCast_gpu.cu $(CytnxPATH)/src/utils/utils_internal_gpu/cuCast_gpu.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuMovemem_gpu.o: $(CytnxPATH)/src/utils/utils_internal_gpu/cuMovemem_gpu.cu $(CytnxPATH)/src/utils/utils_internal_gpu/cuMovemem_gpu.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuComplexmem_gpu.o: $(CytnxPATH)/src/utils/utils_internal_gpu/cuComplexmem_gpu.cu $(CytnxPATH)/src/utils/utils_internal_gpu/cuComplexmem_gpu.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuSetZeros_gpu.o: $(CytnxPATH)/src/utils/utils_internal_gpu/cuSetZeros_gpu.cu $(CytnxPATH)/src/utils/utils_internal_gpu/cuSetZeros_gpu.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuReduce_gpu.o: $(CytnxPATH)/src/utils/utils_internal_gpu/cuReduce_gpu.cu $(CytnxPATH)/src/utils/utils_internal_gpu/cuReduce_gpu.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuSetArange_gpu.o: $(CytnxPATH)/src/utils/utils_internal_gpu/cuSetArange_gpu.cu $(CytnxPATH)/src/utils/utils_internal_gpu/cuSetArange_gpu.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuGetElems_gpu.o: $(CytnxPATH)/src/utils/utils_internal_gpu/cuGetElems_gpu.cu $(CytnxPATH)/src/utils/utils_internal_gpu/cuGetElems_gpu.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuSetElems_gpu.o: $(CytnxPATH)/src/utils/utils_internal_gpu/cuSetElems_gpu.cu $(CytnxPATH)/src/utils/utils_internal_gpu/cuSetElems_gpu.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuFill_gpu.o: $(CytnxPATH)/src/utils/utils_internal_gpu/cuFill_gpu.cu $(CytnxPATH)/src/utils/utils_internal_gpu/cuFill_gpu.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
endif




## Linalg:
########################
Add.o: $(CytnxPATH)/src/linalg/Add.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Mul.o: $(CytnxPATH)/src/linalg/Mul.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Sub.o: $(CytnxPATH)/src/linalg/Sub.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Div.o: $(CytnxPATH)/src/linalg/Div.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
iAdd.o: $(CytnxPATH)/src/linalg/iAdd.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
iMul.o: $(CytnxPATH)/src/linalg/iMul.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
iSub.o: $(CytnxPATH)/src/linalg/iSub.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
iDiv.o: $(CytnxPATH)/src/linalg/iDiv.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<


Cpr.o: $(CytnxPATH)/src/linalg/Cpr.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Mod.o: $(CytnxPATH)/src/linalg/Mod.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Svd.o: $(CytnxPATH)/src/linalg/Svd.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Svd_truncate.o: $(CytnxPATH)/src/linalg/Svd_truncate.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
InvM.o: $(CytnxPATH)/src/linalg/InvM.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
InvM_.o: $(CytnxPATH)/src/linalg/InvM_.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Inv_.o: $(CytnxPATH)/src/linalg/Inv_.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Inv.o: $(CytnxPATH)/src/linalg/Inv.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Conj.o: $(CytnxPATH)/src/linalg/Conj.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Conj_.o: $(CytnxPATH)/src/linalg/Conj_.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Exp.o: $(CytnxPATH)/src/linalg/Exp.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Exp_.o: $(CytnxPATH)/src/linalg/Exp_.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Expf.o: $(CytnxPATH)/src/linalg/Expf.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Expf_.o: $(CytnxPATH)/src/linalg/Expf_.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
ExpH.o: $(CytnxPATH)/src/linalg/ExpH.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
ExpM.o: $(CytnxPATH)/src/linalg/ExpM.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Diag.o: $(CytnxPATH)/src/linalg/Diag.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Det.o: $(CytnxPATH)/src/linalg/Det.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Norm.o: $(CytnxPATH)/src/linalg/Norm.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Eigh.o: $(CytnxPATH)/src/linalg/Eigh.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Eig.o: $(CytnxPATH)/src/linalg/Eig.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Qr.o: $(CytnxPATH)/src/linalg/Qr.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Qdr.o: $(CytnxPATH)/src/linalg/Qdr.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Max.o: $(CytnxPATH)/src/linalg/Max.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Min.o: $(CytnxPATH)/src/linalg/Min.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Sum.o: $(CytnxPATH)/src/linalg/Sum.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Matmul.o: $(CytnxPATH)/src/linalg/Matmul.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Matmul_dg.o: $(CytnxPATH)/src/linalg/Matmul_dg.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Tensordot.o: $(CytnxPATH)/src/linalg/Tensordot.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Tensordot_dg.o: $(CytnxPATH)/src/linalg/Tensordot_dg.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Vectordot.o: $(CytnxPATH)/src/linalg/Vectordot.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Outer.o: $(CytnxPATH)/src/linalg/Outer.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Kron.o: $(CytnxPATH)/src/linalg/Kron.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Dot.o: $(CytnxPATH)/src/linalg/Dot.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Trace.o: $(CytnxPATH)/src/linalg/Trace.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Tridiag.o: $(CytnxPATH)/src/linalg/Tridiag.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Pow.o: $(CytnxPATH)/src/linalg/Pow.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Pow_.o: $(CytnxPATH)/src/linalg/Pow_.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Abs.o: $(CytnxPATH)/src/linalg/Abs.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Abs_.o: $(CytnxPATH)/src/linalg/Abs_.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Hosvd.o: $(CytnxPATH)/src/linalg/Hosvd.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Lanczos_ER.o: $(CytnxPATH)/src/linalg/Lanczos_ER.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Lanczos_Gnd.o: $(CytnxPATH)/src/linalg/Lanczos_Gnd.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Lanczos_Gnd_Ut.o: $(CytnxPATH)/src/linalg/Lanczos_Gnd_Ut.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Lstsq.o: $(CytnxPATH)/src/linalg/Lstsq.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<


test.o: test.cpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
dmrg_tfim.o: dmrg_tfim.cpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
#dmrg_tfim2.o: dmrg_tfim2.cpp
#	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
#ed_ising.o: ed_ising.cpp
#	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

.phony : clean cleanpy

clean:
	rm *.o test *.so 

cleanpy:
	rm ./cytnx/*.so
