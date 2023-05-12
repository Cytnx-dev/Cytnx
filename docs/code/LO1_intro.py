import sys
sys.path.append("../../Cytnx")
import cytnx
import numpy as np 
x = cytnx.ones(4)
H = cytnx.arange(16).reshape(4,4)

y = cytnx.linalg.Dot(H,x)

print(x)
print(H)
print(y)

