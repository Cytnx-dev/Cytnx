T = cytnx.UniTensor(cytnx.arange(12).reshape(4,3))
T.reshape_(2,3,2)
T.print_diagram()
