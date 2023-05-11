import sys
sys.path.append("../../Cytnx")
import cytnx
import numpy as np 

A = cytnx.arange(12).reshape(3,4)
A.Save("T1")

B = cytnx.Tensor.Load("T1.cytn")
print(B)

