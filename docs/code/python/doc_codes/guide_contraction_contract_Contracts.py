# Create A1, A2, M
A1 = cytnx.UniTensor.normal([2,8,8], mean=0., std=1., dtype=cytnx.Type.ComplexDouble) \
                    .set_name("A1")

A2 = A1.Conj().set_name("A2")
M = cytnx.UniTensor.ones([2,2,4,4]).set_name("M")

# Assign labels
A1.relabel_(["phy1","v1","v2"])
M.relabel_(["phy1","phy2","v3","v4"])
A2.relabel_(["phy2","v5","v6"])

# Use Contract
Res = cytnx.Contract([A1,M,A2], order = "(M,(A1,A2))", optimal = False) \
           .set_name("Res")
Res.print_diagram()
