bd_sym_u1_c = cytnx.Bond(cytnx.BD_KET,\
                [cytnx.Qs(-1)>>2,cytnx.Qs(1)>>3,cytnx.Qs(2)>>4,cytnx.Qs(-2)>>5,cytnx.Qs(0)>>6])
print(bd_sym_u1_c)

bd_sym_all = bd_sym_u1_a.combineBond(bd_sym_u1_c)
print(bd_sym_all)
