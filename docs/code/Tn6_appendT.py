import sys
sys.path.append("../../Cytnx")
import cytnx
import numpy as np 

A = cytnx.ones([3,4,5])
B = cytnx.ones([4,5])*2
print(A)
print(B)

A.append(B)
print(A)

