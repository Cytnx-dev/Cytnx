# Creating A1, A2, M
A1 = cytnx.UniTensor(
    cytnx.random.normal(
        [2,8,8], mean=0., std=1., 
        dtype=cytnx.Type.ComplexDouble))

A2 = A1.Conj()
M = cytnx.UniTensor(cytnx.ones([2,2,4,4]))

# Calling ncon
Res = cytnx.ncon([A1,M,A2],
                [[1,-1,-2],[1,2,-3,-4],[2,-5,-6]])
Res.print_diagram()
