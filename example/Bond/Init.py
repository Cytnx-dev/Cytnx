import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *

bd_a = Bond(10)
print(bd_a)

bd_b = Bond(10,BD_KET)
print(bd_b)

bd_c = Bond(BD_IN,[Qs(0)>>1,Qs(2)>>1,Qs(-1)>>1,Qs(3)>>1])
print(bd_c)

bd_d = Bond(BD_OUT,[Qs(0 ,0)>>1,
                      Qs(2 ,1)>>1,
                      Qs(-1,1)>>1,
                      Qs(3 ,0)>>1],
                     [Symmetry.U1(),
                      Symmetry.Zn(2)])

print(bd_d)
