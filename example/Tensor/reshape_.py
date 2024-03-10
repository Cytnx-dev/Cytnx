import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *

A = arange(60)
print(A)
A.reshape_(5,12)
print(A)
