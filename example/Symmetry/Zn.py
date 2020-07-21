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

