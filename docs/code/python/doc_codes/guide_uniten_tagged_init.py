bond_a = cytnx.Bond(3, cytnx.BD_KET)
bond_b = cytnx.Bond(3, cytnx.BD_BRA)
Ta = cytnx.UniTensor([bond_a, bond_a, bond_b], 
                     labels=["a", "b", "c"], 
                     rowrank = 2)
Ta.set_name("Ta")
Ta.print_diagram()
Tb = cytnx.UniTensor([bond_a, bond_b, bond_b], 
                     labels=["c", "d", "e"], 
                     rowrank = 1)
Tb.set_name("Tb")
Tb.print_diagram()
Tc = cytnx.UniTensor([bond_b, bond_b, bond_b], 
                     labels=["c", "d", "e"], 
                     rowrank = 1)
Tc.set_name("Tc")
Tc.print_diagram()
