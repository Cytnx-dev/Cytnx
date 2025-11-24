uT = cytnx.UniTensor.ones([2,3,4]) \
                    .relabel(["a","b","c"]) \
                    .set_name("untagged tensor")

bond_d = cytnx.Bond(
    cytnx.BD_IN, [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],
    [cytnx.Symmetry.U1()])

bond_e = cytnx.Bond(
    cytnx.BD_IN, [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],
    [cytnx.Symmetry.U1()])

bond_f = cytnx.Bond(
    cytnx.BD_OUT,
    [cytnx.Qs(2)>>1, cytnx.Qs(0)>>2,
     cytnx.Qs(-2)>>1],
    [cytnx.Symmetry.U1()])

bond_g = cytnx.Bond(2,cytnx.BD_OUT)
bond_h = cytnx.Bond(2,cytnx.BD_IN)

Tsymm = cytnx.UniTensor([bond_d, bond_e, bond_f]) \
                       .relabel(["d","e","f"]) \
                       .set_name("symm. tensor")

Tdiag = cytnx.UniTensor([bond_g, bond_h], is_diag=True) \
                       .relabel(["g","h"]) \
                       .set_name("diag tensor")