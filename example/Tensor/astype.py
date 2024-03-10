import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *

A = zeros([3,4,5],dtype=Type.Double)
print(A)

B = A.astype(Type.Uint64)
print(B)

C = A.astype(Type.Double)
print(C is A)
