from cytnx import *

bd_a = Bond(10)
print(bd_a)

bd_b = Bond(5)
print(bd_b)

bd_a.combineBond_(bd_b)
print(bd_a)


bd_c = Bond(3,BD_BRA,[[0,2,-4],
                      [1,0,1]],
                  [Symmetry.U1(),
                   Symmetry.Zn(2)]);
print(bd_c)                                

bd_d = Bond(4,BD_BRA,[[0,2,-1,3],
                      [0,1, 1,0]],
                     [Symmetry.U1(),
                      Symmetry.Zn(2)]);
print(bd_d)
    
bd_c.combineBond_(bd_d)
print(bd_c)

