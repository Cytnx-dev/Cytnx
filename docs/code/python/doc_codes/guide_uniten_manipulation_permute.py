bond_d = cytnx.Bond(
    cytnx.BD_IN, 
    [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],
    [cytnx.Symmetry.U1()])

bond_e = cytnx.Bond(
    cytnx.BD_IN, 
    [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],
    [cytnx.Symmetry.U1()])

bond_f = cytnx.Bond(
    cytnx.BD_OUT,
    [cytnx.Qs(2)>>1, cytnx.Qs(0)>>2, 
     cytnx.Qs(-2)>>1],
    [cytnx.Symmetry.U1()])

Tsymm = cytnx.UniTensor(
    [bond_d, bond_e, bond_f], 
    name="symm. tensor", labels=["d","e","f"])

Tsymm.print_diagram()

Tsymm_perm_ind=Tsymm.permute([2,0,1])
Tsymm_perm_ind.print_diagram()

Tsymm_perm_label=Tsymm.permute(["f","d","e"])
Tsymm_perm_label.print_diagram()
