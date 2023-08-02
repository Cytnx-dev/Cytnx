import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *


#1. Create a Tensor with
#  shape (3,4,5),
#  dtype =Type.Double [default],
#  device=Device.cpu [default]
A = Tensor([3,4,5])
print(A)

#2. Create a Tensor with
#  shape (3,4,5),
#  dtype =Type.Uint64,
#  device=Device.cpu [default],
#  [Note] the dtype can be any one of the supported type.
B = Tensor([3,4,5],dtype=Type.Uint64)
print(B)

#3. Initialize a Tensor with
#  shape (3,4,5),
#  dtype =Type.Double,
#  device=Device.cuda+0, (on gpu with gpu-id=0)
#  [Note] the gpu device can be set with Device.cuda+<gpu-id>
C = Tensor([3,4,5],dtype=Type.Double,device=Device.cuda+0);
print(C)

#4. Create an empty Tensor, and init later
D = Tensor()
D.Init([3,4,5],dtype=Type.Double,device=Device.cpu);
