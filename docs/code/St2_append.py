import sys
sys.path.append("../../Cytnx")
import cytnx
import numpy as np 
A = cytnx.Storage(4)
A.set_zeros();
print(A)

A.append(500)
print(A)
print(A.capacity())
