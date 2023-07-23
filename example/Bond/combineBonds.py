import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *

bd_a = Bond(10,BD_KET);
bd_b = Bond(4,BD_KET);
bd_c = Bond(5,BD_KET);
bd_d = Bond(2,BD_KET);

bd_all = bd_a.combineBonds([bd_b,bd_c,bd_d]);

print( bd_a )
print( bd_b )
print( bd_c )
print( bd_d )
print( bd_all )


bd_sym_a = Bond(BD_BRA,[Qs(0,1)>>1,
                        Qs(2,0)>>1,
                        Qs(-4,1)>>1],
                          [Symmetry.U1(),
                           Symmetry.Zn(2)]);

bd_sym_b = Bond(BD_BRA,[Qs(0 ,0)>>1,
                        Qs(2 ,1)>>1,
                        Qs(-1,1)>>1,
                        Qs(3 ,0)>>1],
                          [Symmetry.U1(),
                           Symmetry.Zn(2)]);

bd_sym_c = Bond(BD_BRA,[Qs(1 ,1)>>2,
                        Qs(-1,1)>>1,
                        Qs(-2,0)>>1,
                        Qs(0 ,0)>>1],
                          [Symmetry.U1(),
                           Symmetry.Zn(2)]);

bd_sym_d = bd_sym_a.combineBonds([bd_sym_b,bd_sym_c]);
print( bd_sym_a )
print( bd_sym_b )
print( bd_sym_c )
print( bd_sym_d )
