CYTNX_ROOT=../
CC := g++
CCFLAGS := -std=c++11 
INCFLAGS := $(CCFLAGS) -I$(CYTNX_ROOT)/include
LDFLAGS := -L$(CYTNX_ROOT)
SO := $(CYTNX_ROOT)/libcytnx.so
#SO := -lcytnx


 
Tensor_example = Tensor_Init.e Tensor_to.e Tensor_to_.e Tensor_clone.e Tensor_permute.e Tensor_contiguous.e Tensor_contiguous_.e Tensor_reshape.e Tensor_reshape_.e Tensor_at.e Tensor_item.e Tensor_astype.e Tensor_get.e Tensor_set.e Tensor_fill.e
Storage_example = Storage_Init.e Storage_astype.e Storage_eq.e Storage_clone.e Storage_real.e Storage_imag.e
Bond_example = Bond_Init.e Bond_clone.e Bond_combineBondinplace.e Bond_combineBond.e Bond_combineBonds.e Bond_combineBonds_.e
Accessor_example = Accessor_ex.e
Symmetry_example = Symmetry_U1.e Symmetry_Zn.e Symmetry_clone.e
Network_example = Network_Fromfile.e Network_FromString.e 
UniTensor_example = UniTensor_fromTensor.e 
all: $(Tensor_example) $(Storage_example) $(Bond_example) $(Accessor_example) $(Symmetry_example) $(Network_example) $(UniTensor_example)


Tensor_Init.e: ./Tensor/Init.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(SO)
Tensor_to.e: ./Tensor/to.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(SO)	
Tensor_to_.e: ./Tensor/to_.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(SO)
Tensor_clone.e: ./Tensor/clone.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(SO)	
Tensor_permute.e: ./Tensor/permute.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(SO)
Tensor_contiguous.e: ./Tensor/contiguous.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(SO)	
Tensor_contiguous_.e: ./Tensor/contiguous_.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< 	$(SO)
Tensor_reshape_.e: ./Tensor/reshape_.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< 	$(SO)
Tensor_reshape.e: ./Tensor/reshape.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< 	$(SO)
Tensor_at.e: ./Tensor/at.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< 	$(SO)
Tensor_item.e: ./Tensor/item.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< 	$(SO)
Tensor_astype.e: ./Tensor/astype.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< 	$(SO)
Tensor_get.e: ./Tensor/get.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< 	$(SO)
Tensor_set.e: ./Tensor/set.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< 	$(SO)
Tensor_fill.e: ./Tensor/fill.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< 	$(SO)

Storage_Init.e: ./Storage/Init.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< 	$(SO)

Storage_astype.e: ./Storage/astype.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(SO) 	

Storage_eq.e: ./Storage/eq.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< 	$(SO)

Storage_clone.e: ./Storage/clone.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< 	$(SO)

Storage_real.e: ./Storage/real.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< 	$(SO)

Storage_imag.e: ./Storage/imag.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< 	$(SO)

Bond_Init.e: ./Bond/Init.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $<  $(SO)
Bond_clone.e: ./Bond/clone.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $<  $(SO)
Bond_combineBondinplace.e: ./Bond/combineBondinplace.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $<  $(SO)
Bond_combineBond.e: ./Bond/combineBond.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $<  $(SO)
Bond_combineBonds.e: ./Bond/combineBonds.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $<  $(SO)
Bond_combineBonds_.e: ./Bond/combineBonds_.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $<  $(SO)

Accessor_ex.e: ./Accessor/example.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(SO) 

Symmetry_U1.e: ./Symmetry/U1.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(SO)
Symmetry_Zn.e: ./Symmetry/Zn.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(SO)
Symmetry_clone.e: ./Symmetry/clone.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $<  $(SO)

Network_Fromfile.e: ./Network/Fromfile.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $<  $(SO)
Network_FromString.e: ./Network/FromString.cpp 
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $<  $(SO)
UniTensor_fromTensor.e: ./UniTensor/fromTensor.cpp
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $<  $(SO)


.phony: clean

clean:
	rm $(Tensor_example) $(Storage_example) $(Bond_example) $(Accessor_example) $(Symmetry_example) $(Network_example) $(UniTensor_example) 


