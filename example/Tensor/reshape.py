import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *

A = arange(60)

B = A.reshape(5,12)
print(A)
print(B)
