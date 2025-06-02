import sys
sys.path.append("../../Cytnx")
import cytnx
import numpy as np 

A = cytnx.ones(4)
print(A)
A.append(4)
print(A)

