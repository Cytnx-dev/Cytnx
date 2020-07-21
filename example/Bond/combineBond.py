from cytnx import *

bd_a = Bond(10,BD_KET)
bd_b = Bond(15,BD_KET)
bd_c = bd_a.combineBond(bd_b)
print(bd_a)
print(bd_b)
print(bd_c)


bd_d = Bond(3,BD_BRA,[[0,1],[2,0],[-4,1]],
                      [Symmetry.U1(),
                       Symmetry.Zn(2)])
                                
bd_e = Bond(4,BD_BRA,[[0,0],[2,1],[-1,1],[3,0]],
                      [Symmetry.U1(),
                       Symmetry.Zn(2)])

bd_f = bd_d.combineBond(bd_e)
print(bd_f)
print(bd_d)
print(bd_e)
