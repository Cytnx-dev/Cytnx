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
INCFLAGS :=-I$(CytnxPATH)/include -$=I$(CytnxPATH)/src


ifeq ($(ICPC_Enable),1)
  CC:= $(ICPC)
  CCFLAGS:= -DUNI_ICPC
else
  CC:= $(GCC)
  CCFLAGS:=
endif

ifeq ($(MKL_Enable),1)
  CCFLAGS += -std=c++11 -g -Wformat=0 -fPIC -DUNI_MKL
  LDFLAGS += $(DOCKER_MKL) -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -ldl -lstdc++
else
  CCFLAGS += -std=c++11 -g -Wformat=0 -fPIC 
  LDFLAGS +=  -llapack -lblas -lstdc++
endif


NVCC:= $(CUDA_PATH)/bin/nvcc -ccbin $(CC)
NVFLAGS:= -m64 -g
SMS ?= 30
GENCODE_FLAGS:= -arch=sm_$(SMS)

ifeq ($(OMP_Enable),1)
  CCFLAGS += -fopenmp -DUNI_OMP
endif

ifeq ($(DEBUG_Enable),1)
  CCFLAGS += -DUNI_DEBUG
endif

ALL_CCFLAGS := 
ifeq ($(GPU_Enable),1)
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
  LDFLAGS += -lcublas -lcusolver -lcurand
  ALL_LDFLAGS += $(addprefix -Xlinker , $(LDFLAGS))
  ALL_LDFLAGS += -L$(CUDA_PATH)/lib64
  LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart
else
  ALL_LDFLAGS += $(LDFLAGS)
endif

OBJS = Storage_base.o BoolStorage.o Uint16Storage.o Int16Storage.o Uint32Storage.o Int32Storage.o Uint64Storage.o Int64Storage.o FloatStorage.o DoubleStorage.o ComplexFloatStorage.o ComplexDoubleStorage.o Type.o Device.o


OBJS += Network.o Network_base.o RegularNetwork.o FermionNetwork.o Storage.o Bond.o Tensor.o Symmetry.o Accessor.o Generator.o UniTensor_base.o DenseUniTensor.o SparseUniTensor.o UniTensor.o contraction_tree.o



## Utils
OBJS += utils_internal_interface.o
OBJS += utils.o Cast_cpu.o Alloc_cpu.o Movemem_cpu.o vec_range.o complex_arithmetic.o is.o vec_intersect.o vec_concatenate.o vec_where.o vec_erase.o vec_clone.o vec_unique.o vec_map.o SetZeros_cpu.o Fill_cpu.o SetArange_cpu.o GetElems_cpu.o SetElems_cpu.o cartesian.o str_utils.o
ifeq ($(GPU_Enable),1)
  OBJS += cucomplex_arithmetic.o cuAlloc_gpu.o cuCast_gpu.o cuMovemem_gpu.o cuSetZeros_gpu.o cuFill_gpu.o cuSetArange_gpu.o cuGetElems_gpu.o  cuSetElems_gpu.o
endif

## Linalg_internal
OBJS += linalg_internal_interface.o

OBJS += Cpr_internal.o Add_internal.o Sub_internal.o Mul_internal.o Div_internal.o Arithmetic_internal.o Svd_internal.o Inv_inplace_internal.o Conj_inplace_internal.o Exp_internal.o Eigh_internal.o Matmul_internal.o Diag_internal.o Outer_internal.o Vectordot_internal.o 

ifeq ($(GPU_Enable),1)
  OBJS += cuCpr_internal.o cuAdd_internal.o cuSub_internal.o cuMul_internal.o cuDiv_internal.o cuArithmetic_internal.o cuSvd_internal.o cuInv_inplace_internal.o cuConj_inplace_internal.o cuExp_internal.o  cuEigh_internal.o cuMatmul_internal.o cuDiag_internal.o #cuOuter_internal.o

endif

## Linalg
OBJS += Add.o Div.o Sub.o Mul.o Cpr.o Svd.o Svd_truncate.o Inv.o Inv_.o Conj.o Conj_.o Exp.o Exp_.o Eigh.o Diag.o Matmul.o Tensordot.o Otimes.o Vectordot.o Tridiag.o


ALLOBJS = $(OBJS)
##
ifeq ($(GPU_Enable),1)
  ALLOBJS += gpusrc.o
endif

TESTPATH=tests


all: test demo 


#test: test.o $(ALLOBJS)
#	$(CC) -o $@ $^ $(CCFLAGS) $(LDFLAGS)

test: test.o libcytnx.so
	$(CC) -L. $(LDFLAGS) -o $@ $< libcytnx.so
	#export LD_LIBRARY_PATH=.

demo: demo.o libcytnx.so
	$(CC) -L. $(LDFLAGS) -o $@ $< libcytnx.so
	#export LD_LIBRARY_PATH=.

libcytnx.so: $(ALLOBJS)
	$(CC) -shared -o $@ $^ $(CCFLAGS) $(LDFLAGS)

pyobj: $(ALLOBJS)
	$(CC) $(INCFLAGS) $(CCFLAGS) $(PYOBJFLAGS) $(shell python3 -m pybind11 --includes)  pybind/cytnx.cpp $^ $(LDFLAGS) -shared -o cytnx/cytnx$(shell python3-config --extension-suffix)


doc : 
	doxygen docs.doxygen


ifeq ($(GPU_Enable),1)
gpusrc.o: $(OBJS)
	$(NVCC) $(ALL_CCFLAGS) $(ALL_LDFLAGS) -dlink $^ -o $@ 
endif

## Tensor
Tensor.o: $(CytnxPATH)/src/Tensor.cpp $(CytnxPATH)/include/Tensor.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Bond.o : $(CytnxPATH)/src/Bond.cpp $(CytnxPATH)/include/Bond.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Symmetry.o: $(CytnxPATH)/src/Symmetry.cpp $(CytnxPATH)/include/Symmetry.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Storage.o: $(CytnxPATH)/src/Storage.cpp $(CytnxPATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Accessor.o: $(CytnxPATH)/src/Accessor.cpp $(CytnxPATH)/include/Accessor.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Generator.o: $(CytnxPATH)/src/Generator.cpp $(CytnxPATH)/include/Generator.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

UniTensor.o: $(CytnxPATH)/src/UniTensor.cpp $(CytnxPATH)/include/UniTensor.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<


UniTensor_base.o: $(CytnxPATH)/src/UniTensor_base.cpp $(CytnxPATH)/include/UniTensor.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
DenseUniTensor.o: $(CytnxPATH)/src/DenseUniTensor.cpp $(CytnxPATH)/include/UniTensor.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
SparseUniTensor.o: $(CytnxPATH)/src/SparseUniTensor.cpp $(CytnxPATH)/include/UniTensor.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<


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

## Network:
Network.o: $(CytnxPATH)/src/Network.cpp $(CytnxPATH)/include/Network.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Network_base.o: $(CytnxPATH)/src/Network_base.cpp $(CytnxPATH)/include/Network.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

RegularNetwork.o: $(CytnxPATH)/src/RegularNetwork.cpp $(CytnxPATH)/include/Network.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

FermionNetwork.o: $(CytnxPATH)/src/FermionNetwork.cpp $(CytnxPATH)/include/Network.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  



## linalg_internal
###########################
linalg_internal_interface.o : $(CytnxPATH)/src/linalg/linalg_internal_interface.cpp $(CytnxPATH)/src/linalg/linalg_internal_interface.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Arithmetic_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Arithmetic_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Arithmetic_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Cpr_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Cpr_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Cpr_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Add_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Add_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Add_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Mul_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Mul_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Mul_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Sub_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Sub_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Sub_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  


Div_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Div_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Div_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Svd_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Svd_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Svd_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Eigh_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Eigh_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Eigh_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Exp_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Exp_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Exp_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Diag_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Diag_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Diag_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Inv_inplace_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Inv_inplace_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Inv_inplace_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Conj_inplace_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Conj_inplace_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Conj_inplace_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Matmul_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Matmul_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Matmul_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Outer_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Outer_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Outer_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Vectordot_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Vectordot_internal.cpp $(CytnxPATH)/src/linalg/linalg_internal_cpu/Vectordot_internal.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

Tridiag_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_cpu/Tridiag_internal.cpp $(CytnxPATH)/include/linalg/linalg_internal_cpu/Tridiag_internal.hpp
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

cuSvd_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuSvd_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuSvd_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuEigh_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuEigh_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuEigh_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuExp_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuExp_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuExp_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuDiag_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuDiag_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuDiag_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuInv_inplace_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuInv_inplace_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuInv_inplace_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuConj_inplace_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuConj_inplace_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuConj_inplace_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuMatmul_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuMatmul_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuMatmul_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@

cuOuter_internal.o :  $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuOuter_internal.cu $(CytnxPATH)/src/linalg/linalg_internal_gpu/cuOuter_internal.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@


endif



## Misc
##########################
Type.o : $(CytnxPATH)/src/Type.cpp $(CytnxPATH)/include/Type.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Device.o: $(CytnxPATH)/src/Device.cpp $(CytnxPATH)/include/Device.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

contraction_tree.o: $(CytnxPATH)/src/contraction_tree.cpp $(CytnxPATH)/include/contraction_tree.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

## Utils
##########################
utils_internal_interface.o: $(CytnxPATH)/src/utils/utils_internal_interface.cpp $(CytnxPATH)/src/utils/utils_internal_interface.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

utils.o: $(CytnxPATH)/src/utils/utils.cpp $(CytnxPATH)/include/utils/utils.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

Cast_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/Cast_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/Cast_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

Movemem_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/Movemem_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/Movemem_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

Alloc_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/Alloc_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/Alloc_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

SetZeros_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/SetZeros_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/SetZeros_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

SetArange_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/SetArange_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/SetArange_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

GetElems_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/GetElems_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/GetElems_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
SetElems_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/SetElems_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/SetElems_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Fill_cpu.o: $(CytnxPATH)/src/utils/utils_internal_cpu/Fill_cpu.cpp $(CytnxPATH)/src/utils/utils_internal_cpu/Fill_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<


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
vec_concatenate.o: $(CytnxPATH)/src/utils/vec_concatenate.cpp $(CytnxPATH)/include/utils/vec_concatenate.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
vec_intersect.o: $(CytnxPATH)/src/utils/vec_intersect.cpp $(CytnxPATH)/include/utils/vec_intersect.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
str_utils.o: $(CytnxPATH)/src/utils/str_utils.cpp $(CytnxPATH)/include/utils/str_utils.hpp
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
cuSetZeros_gpu.o: $(CytnxPATH)/src/utils/utils_internal_gpu/cuSetZeros_gpu.cu $(CytnxPATH)/src/utils/utils_internal_gpu/cuSetZeros_gpu.hpp
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
Cpr.o: $(CytnxPATH)/src/linalg/Cpr.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Svd.o: $(CytnxPATH)/src/linalg/Svd.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Svd_truncate.o: $(CytnxPATH)/src/linalg/Svd_truncate.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Inv.o: $(CytnxPATH)/src/linalg/Inv.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Inv_.o: $(CytnxPATH)/src/linalg/Inv_.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Conj.o: $(CytnxPATH)/src/linalg/Conj.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Conj_.o: $(CytnxPATH)/src/linalg/Conj_.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Exp.o: $(CytnxPATH)/src/linalg/Exp.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Exp_.o: $(CytnxPATH)/src/linalg/Exp_.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Diag.o: $(CytnxPATH)/src/linalg/Diag.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Eigh.o: $(CytnxPATH)/src/linalg/Eigh.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Matmul.o: $(CytnxPATH)/src/linalg/Matmul.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Tensordot.o: $(CytnxPATH)/src/linalg/Tensordot.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Vectordot.o: $(CytnxPATH)/src/linalg/Vectordot.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Otimes.o: $(CytnxPATH)/src/linalg/Otimes.cpp $(CytnxPATH)/include/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<
Tridiag.o: $(CytnxPATH)/src/linalg/Tridiag.cpp $(CytnxPATH)/include/linalg/linalg.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

test.o: $(TESTPATH)/test.cpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<
demo.o: demo.cpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

.phony : clean cleanpy

clean:
	rm *.o test *.so 

cleanpy:
	rm ./cytnx/*.so
