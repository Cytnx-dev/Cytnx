from cytnx import Bond, BD_IN, BD_OUT, Qs, Symmetry
# bond1 = Bond(BD_IN,[[2,0], [4,1]],[3,5],[Symmetry.U1(), Symmetry.Zn(2)])
# bond2 = Bond(BD_IN,[Qs(2,0)>>3, Qs(4,1)>>5],[Symmetry.U1(), Symmetry.Zn(2)])
bd1 = cytnx.Bond(cytnx.BD_IN,[[1],[-1]],[1,1])
bd2 = cytnx.Bond(cytnx.BD_IN,[[1],[-1]],[1,1])
bd3 = cytnx.Bond(cytnx.BD_OUT,[[2],[0],[0],[-2]],[1,1,1,1])

ut = cytnx.UniTensor([bd1,bd2,bd3],rowrank=2)
print(ut)

ut.combineBonds([0,1])
print(ut)
