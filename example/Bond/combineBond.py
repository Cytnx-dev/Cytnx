import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *

bd_a = Bond(10,BD_KET)
bd_b = Bond(15,BD_KET)
bd_c = bd_a.combineBond(bd_b)
print(bd_a)
print(bd_b)
print(bd_c)


bd_d = Bond(BD_BRA,[Qs(0,1)>>1,Qs(2,0)>>1,Qs(-4,1)>>1],
                      [Symmetry.U1(),
                       Symmetry.Zn(2)])

bd_e = Bond(BD_BRA,[Qs(0,0)>>1,Qs(2,1)>>1,Qs(-1,1)>>1,Qs(3,0)>>1],
                      [Symmetry.U1(),
                       Symmetry.Zn(2)])

bd_f = bd_d.combineBond(bd_e)
print(bd_f)
print(bd_d)
print(bd_e)
