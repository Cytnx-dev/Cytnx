CYTNX_ROOT=..
CC := g++
CCFLAGS := -std=c++11 
INCFLAGS := -I$(CYTNX_ROOT)/include
LDFLAGS := -L$(CYTNX_ROOT)

Tensor_example = Tensor_Init.e Tensor_to.e Tensor_to_.e Tensor_clone.e
Storage_example = Storage_Init.e Storage_astype.e Storage_eq.e Storage_clone.e
Bond_example = Bond_Init.e
all: $(Tensor_example) $(Storage_example) $(Bond_example)


Tensor_Init.e: ./Tensor/Init.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	

Tensor_to.e: ./Tensor/to.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	

Tensor_to_.e: ./Tensor/to_.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	

Tensor_clone.e: ./Tensor/clone.cpp $(CYTNX_ROOT)/libcytnx.so
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



.phony: clean

clean:
	rm $(Tensor_example) $(Storage_example) $(Bond_example)


