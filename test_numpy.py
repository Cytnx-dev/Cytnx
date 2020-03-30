import cytnx
import numpy as np

A = cytnx.random.normal(12,0,0.2).reshape(3,4)
B = A.numpy()
print(B)


x = np.arange(100).reshape(10,10)

print(x.dtype)
C = cytnx.from_numpy(x)
print(x)
print(C)
