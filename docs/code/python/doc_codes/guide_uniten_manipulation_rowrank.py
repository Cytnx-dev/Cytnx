# set the rowrank to be 2.
uT = cytnx.UniTensor.ones([5,5,5,5,5]).set_rowrank(2) \
         .relabel(["a", "b", "c", "d", "e"]) \
         .set_name("uT")
uT.print_diagram()

uT.set_rowrank_(3)  # modify the rowrank.
uT.print_diagram()
