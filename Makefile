Tor10PATH=.
CUDA_PATH=/usr/local/cuda
INCFLAGS :=-I$(Tor10PATH)/include

CC:= g++-6
CCFLAGS := -std=c++11 -g -Wformat=0 -fPIC
LDFLAGS :=  -llapack -lblas


PY_Enable=1
GPU_Enable=0
OMP_Enable=1
DEBUG_Enable=1


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

OBJS = Storage_base.o Uint32Storage.o Int32Storage.o Uint64Storage.o Int64Storage.o FloatStorage.o DoubleStorage.o ComplexFloatStorage.o ComplexDoubleStorage.o Type.o Device.o


OBJS += Storage.o Bond.o Tensor.o Symmetry.o

## Utils
OBJS += Cast_cpu.o Alloc_cpu.o Movemem_cpu.o Range_cpu.o
ifeq ($(GPU_Enable),1)
  OBJS += cuAlloc_gpu.o cuCast_gpu.o cuMovemem_gpu.o
endif


ALLOBJS = $(OBJS)
##
ifeq ($(GPU_Enable),1)
  ALLOBJS += gpusrc.o
endif




all: test 

ifeq ($(PY_Enable),1)
  all += pyobj
endif


test: test.o $(ALLOBJS)
	$(CC) $^ $(CCFLAGS) $(LDFLAGS) -o $@

pyobj: $(ALLOBJS)
	$(CC) $(CCFLAGS) $(LDFLAGS) $^ -shared $(shell python3 -m pybind11 --includes) pybind/pytor10.cpp -o pytor10$(shell python3-config --extension-suffix)




ifeq ($(GPU_Enable),1)
gpusrc.o: $(OBJS)
	$(NVCC) $(ALL_CCFLAGS) $(ALL_LDFLAGS) -dlink $^ -o $@ 
endif

## Tensor
Tensor.o: $(Tor10PATH)/src/Tensor.cpp $(Tor10PATH)/include/Tensor.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Bond.o : $(Tor10PATH)/src/Bond.cpp $(Tor10PATH)/include/Bond.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Symmetry.o: $(Tor10PATH)/src/Symmetry.cpp $(Tor10PATH)/include/Symmetry.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Storage.o: $(Tor10PATH)/src/Storage.cpp $(Tor10PATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<


## Storage 
###########################
Storage_base.o: $(Tor10PATH)/src/Storage_base.cpp $(Tor10PATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Uint32Storage.o: $(Tor10PATH)/src/Uint32Storage.cpp $(Tor10PATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Int32Storage.o: $(Tor10PATH)/src/Int32Storage.cpp $(Tor10PATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Uint64Storage.o: $(Tor10PATH)/src/Uint64Storage.cpp $(Tor10PATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
Int64Storage.o: $(Tor10PATH)/src/Int64Storage.cpp $(Tor10PATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  
FloatStorage.o: $(Tor10PATH)/src/FloatStorage.cpp $(Tor10PATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

DoubleStorage.o: $(Tor10PATH)/src/DoubleStorage.cpp $(Tor10PATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

ComplexFloatStorage.o: $(Tor10PATH)/src/ComplexFloatStorage.cpp $(Tor10PATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  

ComplexDoubleStorage.o: $(Tor10PATH)/src/ComplexDoubleStorage.cpp $(Tor10PATH)/include/Storage.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<  


## linalg_internal
###########################
#linalg_internal_interface.o : $(Tor10PATH)/src/linalg_internal_interface.cpp $(Tor10PATH)/include/linalg_internal_interface.hpp

#	$(CC) $(INCFLAGS) -c $<


## Misc
##########################
Type.o : $(Tor10PATH)/src/Type.cpp $(Tor10PATH)/include/Type.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<

Device.o: $(Tor10PATH)/src/Device.cpp $(Tor10PATH)/include/Device.hpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<


## Utils
##########################
Cast_cpu.o: $(Tor10PATH)/src/utils/utils_internal_cpu/Cast_cpu.cpp $(Tor10PATH)/include/utils/utils_internal_cpu/Cast_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

Movemem_cpu.o: $(Tor10PATH)/src/utils/utils_internal_cpu/Movemem_cpu.cpp $(Tor10PATH)/include/utils/utils_internal_cpu/Movemem_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

Alloc_cpu.o: $(Tor10PATH)/src/utils/utils_internal_cpu/Alloc_cpu.cpp $(Tor10PATH)/include/utils/utils_internal_cpu/Alloc_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

Range_cpu.o: $(Tor10PATH)/src/utils/utils_internal_cpu/Range_cpu.cpp $(Tor10PATH)/include/utils/utils_internal_cpu/Range_cpu.hpp
	$(CC)  $(CCFLAGS) $(INCFLAGS) -c $<

ifeq ($(GPU_Enable),1)
cuAlloc_gpu.o: $(Tor10PATH)/src/utils/utils_internal_gpu/cuAlloc_gpu.cu $(Tor10PATH)/include/utils/utils_internal_gpu/cuAlloc_gpu.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuCast_gpu.o: $(Tor10PATH)/src/utils/utils_internal_gpu/cuCast_gpu.cu $(Tor10PATH)/include/utils/utils_internal_gpu/cuCast_gpu.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
cuMovemem_gpu.o: $(Tor10PATH)/src/utils/utils_internal_gpu/cuMovemem_gpu.cu $(Tor10PATH)/include/utils/utils_internal_gpu/cuMovemem_gpu.hpp
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@
endif





test.o: test.cpp
	$(CC) $(CCFLAGS) $(INCFLAGS) -c $<



.phony : clean

clean:
	rm *.o test
