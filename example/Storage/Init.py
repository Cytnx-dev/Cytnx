import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *

A = Storage(10);
print(A)

B = Storage(10,dtype=Type.Uint64)
print(B)

C = Storage(10,device=Device.cuda+0)
print(C)

D = Storage()
D.Init(10,dtype=Type.Double,device=Device.cpu)
