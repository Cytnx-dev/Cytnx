A = cytnx.UniTensor(cytnx.ones([2,8,8]));
A.relabels_(["phy", "left", "right"])
B = cytnx.UniTensor(cytnx.ones([2,8,8]));
B.relabels_(["phy", "left", "right"])
