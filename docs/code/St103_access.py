import sys
sys.path.append("../../Cytnx")
import cytnx
import numpy as np 

A = cytnx.Storage(6)
A.set_zeros()
print(A)

A[4] = 4
print(A)

