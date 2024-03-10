import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *


A = Storage(10)
print(A.dtype_str())

B = A.astype(Type.Double)
print(B.dtype_str())
print(B is A)

C = A.astype(Type.Float)
print(C.dtype_str())
print(C is A)

D = Storage(10,device=Device.cuda+0);
E = D.astype(Type.Float)
print(E.device_str())
