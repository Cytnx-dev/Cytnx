from cytnx import *

A = Storage(15)

B = A
C = A.clone()

print(B is A)
print(C is A)

