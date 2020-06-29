import sys
sys.path.append("../../Cytnx")
import cytnx
import numpy as np 
A = cytnx.Storage(10,dtype=cytnx.Type.Double,device=cytnx.Device.cpu)
A.set_zeros();

print(A);

