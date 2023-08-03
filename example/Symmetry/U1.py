import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *

sym_u1 = Symmetry.U1();

bd_sym_u1_a = Bond(4,BD_KET,[[0],[-4],[-2],[3]],[sym_u1])
bd_sym_u1_b = Bond(4,BD_KET,[[0],[-4],[-2],[3]]) #default is U1 symmetry
print(bd_sym_u1_a)
print(bd_sym_u1_b)
print(bd_sym_u1_a == bd_sym_u1_b,flush=True) #true

bd_sym_u1_c = Bond(5,BD_KET,[[-1],[1],[2],[-2],[0]])
print(bd_sym_u1_c)

bd_sym_all = bd_sym_u1_a.combineBond(bd_sym_u1_c)
print(bd_sym_all)


new_bd_sym_u1_a = Bond(BD_KET,[[0],[-4],[-2],[3]], [1,1,1,1],[sym_u1])
new_bd_sym_u1_b = Bond(BD_KET,[[0],[-4],[-2],[3]], [1,1,1,1]) #default is U1 symmetry
print(new_bd_sym_u1_a)
print(new_bd_sym_u1_b)
print(new_bd_sym_u1_a == new_bd_sym_u1_b,flush=True) #true

new_bd_sym_u1_c = Bond(BD_KET,[[-1],[1],[2],[-2],[0]], [1,1,1,1,1])
print(new_bd_sym_u1_c)

new_bd_sym_all = new_bd_sym_u1_a.combineBond(new_bd_sym_u1_c)
print(new_bd_sym_all)
