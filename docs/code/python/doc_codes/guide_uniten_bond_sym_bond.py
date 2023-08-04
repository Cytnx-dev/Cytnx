# This creates an KET (IN) Bond with quantum number 0,-4,-2,3 with degs 3,4,3,2 respectively.
bd_sym_u1_a = cytnx.Bond(cytnx.BD_KET,\
                        [cytnx.Qs(0)>>3,cytnx.Qs(-4)>>4,cytnx.Qs(-2)>>3,cytnx.Qs(3)>>2],\
                        [cytnx.Symmetry.U1()])

# equivalent:
bd_sym_u1_a = cytnx.Bond(cytnx.BD_IN,\
                        [cytnx.Qs(0),cytnx.Qs(-4),cytnx.Qs(-2),cytnx.Qs(3)],\
                        [3,4,3,2],[cytnx.Symmetry.U1()])

print(bd_sym_u1_a)
