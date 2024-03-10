import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *
bd_a = Bond(10)
print(bd_a)

bd_b = Bond(5)
print(bd_b)

bd_a.combineBond_(bd_b)
print(bd_a)


bd_c = Bond(BD_BRA,[Qs(0,1)>>1,Qs(2,0)>>1,Qs(-4,1)>>1],
                      [Symmetry.U1(),
                       Symmetry.Zn(2)])

print(bd_c)

bd_d = Bond(BD_BRA,[Qs(0,0)>>1,Qs(2,1)>>1,Qs(-1,1)>>1,Qs(3,0)>>1],
                      [Symmetry.U1(),
                       Symmetry.Zn(2)])
print(bd_d)

bd_c.combineBond_(bd_d)
print(bd_c)
