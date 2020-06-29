import sys
sys.path.append("../../Cytnx")
import cytnx

A = cytnx.ones([2,2]); #on CPU
B = A.to(cytnx.Device.cuda+0);
print(A)
print(B)

A.to_(cytnx.Device.cuda)
print(A)
