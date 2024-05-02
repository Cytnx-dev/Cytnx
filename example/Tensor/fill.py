import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *

A = arange(60).reshape(3,4,5)
print(A)

A.fill(999)
print(A)
