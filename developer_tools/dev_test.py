import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
import cytnx
import numpy as np

uT = cytnx.UniTensor.arange(2*3*4, name="tensor uT")
uT.reshape_(2,3,4);
uT.relabels_(["a","b","c"])
