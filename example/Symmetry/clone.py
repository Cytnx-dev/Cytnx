import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *

sym_A = Symmetry.U1()
sym_C = Symmetry.U1()

sym_D = sym_A
print(sym_D is sym_A) #true
print(sym_D == sym_A) #true
print(sym_C is sym_A) #false
print(sym_C == sym_A) #true
