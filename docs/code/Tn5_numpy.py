import sys
sys.path.append("../../Cytnx")
import cytnx
import numpy as np 

A = cytnx.ones([3,4])
B = A.numpy()
print(A)
print(type(B))
print(B)


#-------------------------
B = np.ones([3,4])
A = cytnx.from_numpy(B)
print(B)
print(A)



