import sys
sys.path.append("../../Cytnx")
import cytnx

A = cytnx.arange(24).reshape(2,3,4)
print(A)

B = A[0,:,1:4:2]
print(B)

C = A[:,1]    
print(C)



