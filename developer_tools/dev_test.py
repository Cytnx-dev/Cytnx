import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
import cytnx
import numpy as np

# uT = cytnx.UniTensor.arange(2*3*4, name="tensor uT")
# uT.reshape_(2,3,4);
# uT.relabels_(["a","b","c"])

# T = cytnx.random.uniform([4,4], low=-1., high=1.)
# print(T)
# import time
# time.sleep(2)
# T2 = cytnx.random.uniform([4,4], low=-1., high=1.)
# print(T2)

T = cytnx.random.normal([4,4], mean=0., std=1.)
print(T)
# import time
# time.sleep(2)
T2 = cytnx.random.normal([4,4], mean=0., std=1.)
print(T2)
