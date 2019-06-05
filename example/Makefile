CYTNX_ROOT=..
CC := g++
CCFLAGS := -std=c++11 
INCFLAGS := -I$(CYTNX_ROOT)/include
LDFLAGS := -L$(CYTNX_ROOT)

Tensor_example = Init_ex_tn to_ex_tn to_i_ex_tn clone_ex_tn
Storage_example = Init_ex_st astype_ex_st
all: $(Tensor_example) $(Storage_example)


Init_ex_tn: ./Tensor/Init.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	

to_ex_tn: ./Tensor/to.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	

to_i_ex_tn: ./Tensor/to_.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	

clone_ex_tn: ./Tensor/clone.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	




Init_ex_st: ./Storage/Init.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	

astype_ex_st: ./Storage/astype.cpp $(CYTNX_ROOT)/libcytnx.so
	$(CC) $(INCFLAGS) $(LDFLAGS) -o $@ $< $(CYTNX_ROOT)/libcytnx.so	


.phony: clean

clean:
	rm $(Tensor_example) $(Storage_example)


