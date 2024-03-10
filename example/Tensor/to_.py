import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *

A = Tensor([3,4,5])

A.to_(Device.cuda+0);
print(A.device_str())
