from cytnx import *

A = arange(24);
A.reshape_(2,3,4);
print(A)


B = A[0,:,0:2:1]
print(B)

