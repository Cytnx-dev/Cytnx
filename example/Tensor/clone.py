from cytnx import *

A = Tensor([3,4,5])

B = A
C = A.clone()

print(B is A)
print(C is A)

