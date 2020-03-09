from cytnx import *
from cytnx.cytnx_extension import *

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


