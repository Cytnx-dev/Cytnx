import sys
sys.path.append("../../Cytnx")
import cytnx

A = cytnx.arange(24)
print(A)
A.reshape_(2,3,4)
print(A)



