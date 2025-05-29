# set the rowrank to be 2.
T = cytnx.UniTensor(cytnx.ones([5,5,5,5,5]), rowrank = 2) 
T.print_diagram()

T.set_rowrank(3)  # modify the rowrank.
T.print_diagram()
