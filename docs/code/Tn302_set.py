import sys
sys.path.append("../../Cytnx")
import cytnx

A = cytnx.arange(24).reshape(2,3,4)
B = cytnx.zeros([3,2])
print(A)
print(B)

A[1,:,::2] = B
print(A)

A[0,::2,2] = 4
print(A)

