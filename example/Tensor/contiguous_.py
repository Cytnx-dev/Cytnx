import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *

A = Tensor([3,4,5])
print(A.shape())

B = A.permute(0,2,1)
print(B.shape())

print(B.is_contiguous()) #false
B.contiguous_()
print(B.is_contiguous()) #true
