import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *


A = Storage(10)
print(A.dtype_str())

B = A
C = A.clone()

print(B == A) # true (share same instance)
print(B is A) # true (share same instance)

print(C == A)  # true (the same content.)
print(C is A)  # false (not share same instance)
