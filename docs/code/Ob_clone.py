import sys
sys.path.append("../../Cytnx")
import cytnx
import numpy as np 

A = cytnx.Tensor([2,3]);
B = A;
C = A.clone();

print(B is A)
print(C is A)

