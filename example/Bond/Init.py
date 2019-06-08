from cytnx import *

bd_a = Bond(10)
print(bd_a)

bd_b = Bond(10,BD_KET)
print(bd_b)

bd_c = Bond(4,BD_KET,[[0,2,-1,3]])
print(bd_c)

bd_d = Bond(4,BD_BRA,[[0,2,-1,3],
                      [0,1, 1,0]],
                     [Symmetry.U1(),
                      Symmetry.Zn(2)])

print(bd_d)

    
