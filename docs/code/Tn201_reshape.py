import sys
sys.path.append("../../Cytnx")
import cytnx

A = cytnx.arange(24)
B = A.reshape(2,3,4)
print(A)
print(B)



