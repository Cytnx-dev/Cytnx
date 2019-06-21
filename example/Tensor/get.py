from cytnx import *

A = arange(60).reshape(3,4,5)
print(A)

B = A[2,:,2:5:1]
print(B)

