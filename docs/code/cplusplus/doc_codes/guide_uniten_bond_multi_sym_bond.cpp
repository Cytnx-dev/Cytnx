auto bd_sym_u1z2_a = cytnx::Bond(cytnx::BD_KET,
                                {cytnx::Qs(0 ,0)>>3,
                                 cytnx::Qs(-4,1)>>4,
                                 cytnx::Qs(-2,0)>>3,
                                 cytnx::Qs(3 ,1)>>2},
                                {cytnx::Symmetry::U1(),cytnx::Symmetry::Zn(2)});

print(bd_sym_u1z2_a);
