import sys
sys.path.append("../../Cytnx")
import cytnx
import numpy as np 
A = cytnx.Storage(10)
A.set_zeros()

B = A.astype(cytnx.Type.ComplexDouble)

print(A)
print(B)
