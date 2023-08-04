# Creating A1, A2, M
A1 = cytnx.UniTensor(cytnx.ones([2,8,8]))
A2 = cytnx.UniTensor(cytnx.ones([2,8,8]))
M = cytnx.UniTensor(cytnx.ones([2,2,4,4]))

# Calling ncon
res = cytnx.ncon([A1,M,A2],[[1,-1,-2],[1,2,-3,-4],[2,-5,-6]])
