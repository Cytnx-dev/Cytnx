import sys
sys.path.append("../../Cytnx")
import cytnx

A = cytnx.arange(24).reshape(2,3,4);
B = A.permute(1,2,0)
print(A)
print(B)



