import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *

A = arange(24);
A.reshape_(2,3,4);
print(A)


B = A[0,:,0:2:1]
print(B)
