import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *

A = ones(1,Type.Uint64)
print(A)

print(A.item())
