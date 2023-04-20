import sys
sys.path.append("../../Cytnx")
import cytnx
import numpy as np 

A = cytnx.Storage(4)
A.fill(6)
A.Save("S1")

A = cytnx.Storage.Load("S1.cyst")
print(A)

