import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *
#example of Z2 symmetry object
#------------------------------------
sym_z2 = Symmetry.Zn(2);

bd_sym_z2_A = Bond(4,BD_KET,[[0],[0],[1],[1]],[sym_z2])
bd_sym_z2_B = Bond(3,BD_KET,[[0],[1],[1]],[sym_z2])
print(bd_sym_z2_A)
print(bd_sym_z2_B)

bd_sym_z2all = bd_sym_z2_A.combineBond(bd_sym_z2_B)
print(bd_sym_z2all)


#example of Z4 symmetry object
#------------------------------------
sym_z4 = Symmetry.Zn(4)

bd_sym_z4_A = Bond(4,BD_KET,[[0],[3],[1],[2]],[sym_z4])
bd_sym_z4_B = Bond(3,BD_KET,[[2],[3],[1]],[sym_z4])
print(bd_sym_z4_A)
print(bd_sym_z4_B)

bd_sym_z4all = bd_sym_z4_A.combineBond(bd_sym_z4_B)
print(bd_sym_z4all)

#bk example of Z2 symmetry object
#------------------------------------
new_sym_z2 = Symmetry.Zn(2);

new_bd_sym_z2_A = Bond(BD_KET,[[0],[0],[1],[1]],[1,1,1,1],[new_sym_z2])
new_bd_sym_z2_B = Bond(BD_KET,[[0],[1],[1]],[1,1,1],[new_sym_z2])
print(new_bd_sym_z2_A)
print(new_bd_sym_z2_B)

new_bd_sym_z2all = new_bd_sym_z2_A.combineBond(new_bd_sym_z2_B)
print(new_bd_sym_z2all)


#bk example of Z4 symmetry object
#------------------------------------
new_sym_z4 = Symmetry.Zn(4)

new_bd_sym_z4_A = Bond(BD_KET,[[0],[3],[1],[2]],[1,1,1,1],[new_sym_z4])
new_bd_sym_z4_B = Bond(BD_KET,[[2],[3],[1]],[1,1,1],[new_sym_z4])
print(new_bd_sym_z4_A)
print(new_bd_sym_z4_B)

new_bd_sym_z4all = new_bd_sym_z4_A.combineBond(new_bd_sym_z4_B)
print(new_bd_sym_z4all)
