## A & B share same memory
A = cytnx.Storage(10)
B = cytnx.Tensor.from_storage(A)

## A & C have different memory
C = cytnx.Tensor.from_storage(A.clone())
