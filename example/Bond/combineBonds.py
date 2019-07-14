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


bd_sym_a = Bond(3,BD_BRA,[[0,1],
                          [2,0],
                          [-4,1]],
                          [Symmetry.U1(),
                           Symmetry.Zn(2)]);
                            
bd_sym_b = Bond(4,BD_BRA,[[0 ,0],
                          [2 ,1],
                          [-1,1],
                          [3 ,0]],
                          [Symmetry.U1(),
                           Symmetry.Zn(2)]);

bd_sym_c = Bond(5,BD_BRA,[[1 ,1],
                          [1 ,1],
                          [-1,1],
                          [-2,0],
                          [0 ,0]],
                          [Symmetry.U1(),
                           Symmetry.Zn(2)]);

bd_sym_d = bd_sym_a.combineBonds([bd_sym_b,bd_sym_c]);
print( bd_sym_a )
print( bd_sym_b )
print( bd_sym_c )
print( bd_sym_d )

