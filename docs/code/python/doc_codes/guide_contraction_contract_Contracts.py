# Create A1, A2, M
A1 = cytnx.UniTensor(
    cytnx.random.normal(
        [2,8,8], mean=0., std=1., 
        dtype=cytnx.Type.ComplexDouble), 
    name = "A1");

A2 = A1.Conj();
A2.set_name("A2");
M = cytnx.UniTensor(cytnx.ones([2,2,4,4]), 
                    name = "M")

# Assign labels
A1.relabels_(["phy1","v1","v2"])
M.relabels_(["phy1","phy2","v3","v4"])
A2.relabels_(["phy2","v5","v6"])

# Use Contracts
Res = cytnx.Contracts(TNs = [A1,M,A2], 
                      order = "(M,(A1,A2))", 
                      optimal = False)
Res.print_diagram()
