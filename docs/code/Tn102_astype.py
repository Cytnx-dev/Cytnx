import sys
sys.path.append("../../Cytnx")
import cytnx

A = cytnx.Tensor([3,4],dtype=cytnx.Type.Int64)
B = A.astype(cytnx.Type.Double)
print(A.dtype_str())
print(B.dtype_str())

