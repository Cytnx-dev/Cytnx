uT=cytnx.UniTensor(cytnx.ones([2,3,4]), name="untagged tensor").relabels(["a","b","c"])
bond_d = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],[cytnx.Symmetry.U1()])
bond_e = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],[cytnx.Symmetry.U1()])
bond_f = cytnx.Bond(cytnx.BD_OUT,\
                    [cytnx.Qs(2)>>1, cytnx.Qs(0)>>2, cytnx.Qs(-2)>>1],[cytnx.Symmetry.U1()])
bond_g = cytnx.Bond(2,cytnx.BD_OUT)
bond_h = cytnx.Bond(2,cytnx.BD_IN)
Tsymm = cytnx.UniTensor([bond_d, bond_e, bond_f], name="symm. tensor").relabels(["d","e","f"])
Tdiag = cytnx.UniTensor([bond_g, bond_h], is_diag=True, name="diag tensor").relabels(["g","h"])
