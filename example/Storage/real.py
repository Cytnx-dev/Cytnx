import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *


S1 = Storage(10,Type.ComplexDouble)

for i in range(10):
    S1[i] = i + 1j*(i+1)

S1r = S1.real()
print(S1)
print(S1r)

S2 = Storage(10,Type.ComplexFloat)

for i in range(10):
    S2[i] = i + 1j*(i+1)

S2r = S2.real()
print(S2)
print(S2r)
