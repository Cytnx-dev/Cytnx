A1 = cytnx.UniTensor(
    cytnx.random.normal(
        [2,8,8], mean=0., std=1., 
        dtype=cytnx.Type.ComplexDouble));

A1.relabels_(["phy","v1","v2"]);
A2 = A1.Conj();
A2.relabels_(["phy*","v1*","v2*"]);
