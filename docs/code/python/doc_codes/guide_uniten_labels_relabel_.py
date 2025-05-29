T = cytnx.arange(2*3*4).reshape(2,3,4)
uT = cytnx.UniTensor(T)

uT.relabel_(1,"xx")
uT.print_diagram()

uT.relabels_(["a","b","c"])
uT.print_diagram()
