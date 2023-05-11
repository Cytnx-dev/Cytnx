import sys
sys.path.append("../../Cytnx")
import cytnx

A = cytnx.ones([3,4])
print(A)

B = A + 4 
print(B)

C = A - 7j # type promotion
print(C)

