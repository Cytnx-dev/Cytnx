import sys
sys.path.append("../../Cytnx")
import cytnx
import numpy as np 

A = cytnx.Tensor([2,3])
B = A

print(B is A) # false

