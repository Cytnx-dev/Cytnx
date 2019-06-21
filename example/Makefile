CYTNX_ROOT=..
CC := g++
CCFLAGS := -std=c++11 
INCFLAGS := -I$(CYTNX_ROOT)/include
LDFLAGS := -L$(CYTNX_ROOT)
 
Tensor_example = Tensor_Init.e Tensor_to.e Tensor_to_.e Tensor_clone.e Tensor_permute.e Tensor_contiguous.e Tensor_contiguous_.e Tensor_reshape.e Tensor_reshape_.e Tensor_at.e Tensor_item.e Tensor_astype.e Tensor_get.e Tensor_set.e Tensor_fill.e
Storage_example = Storage_Init.e Storage_astype.e Storage_eq.e Storage_clone.e
Bond_example = Bond_Init.e Bond_clone.e Bond_combineBondinplace.e Bond_combineBond.e Bond_combineBonds.e Bond_combineBonds_.e
Accessor_example = Accessor_ex.e
Symmetry_example = Symmetry_U1.e Symmetry_Zn.e Symmetry_clone.e
all: $(Tensor_example) $(Storage_example) $(Bond_example) $(Accessor_example) $(Symmetry_example)


Tensor_Init.e: ./Tensor/Init.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	
Tensor_to.e: ./Tensor/to.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	
Tensor_to_.e: ./Tensor/to_.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	
Tensor_clone.e: ./Tensor/clone.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	
Tensor_permute.e: ./Tensor/permute.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	
Tensor_contiguous.e: ./Tensor/contiguous.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	
Tensor_contiguous_.e: ./Tensor/contiguous_.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	
Tensor_reshape_.e: ./Tensor/reshape_.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	
Tensor_reshape.e: ./Tensor/reshape.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	
Tensor_at.e: ./Tensor/at.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	
Tensor_item.e: ./Tensor/item.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	
Tensor_astype.e: ./Tensor/astype.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	
Tensor_get.e: ./Tensor/get.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	
Tensor_set.e: ./Tensor/set.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	
Tensor_fill.e: ./Tensor/fill.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	

Storage_Init.e: ./Storage/Init.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	

Storage_astype.e: ./Storage/astype.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	

Storage_eq.e: ./Storage/eq.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	

Storage_clone.e: ./Storage/clone.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	



Bond_Init.e: ./Bond/Init.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so 
Bond_clone.e: ./Bond/clone.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so 
Bond_combineBondinplace.e: ./Bond/combineBondinplace.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so 
Bond_combineBond.e: ./Bond/combineBond.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so 
Bond_combineBonds.e: ./Bond/combineBonds.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so 
Bond_combineBonds_.e: ./Bond/combineBonds_.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so 

Accessor_ex.e: ./Accessor/example.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so

Symmetry_U1.e: ./Symmetry/U1.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so
Symmetry_Zn.e: ./Symmetry/Zn.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so
Symmetry_clone.e: ./Symmetry/clone.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so 

.phony: clean

clean:
	rm $(Tensor_example) $(Storage_example) $(Bond_example) $(Accessor_example) $(Symmetry_example)


