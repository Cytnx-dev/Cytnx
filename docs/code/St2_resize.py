import sys
sys.path.append("../../Cytnx")
import cytnx
import numpy as np 


A = cytnx.Storage(4);
print(A.size());

A.resize(5);
print(A.size());

