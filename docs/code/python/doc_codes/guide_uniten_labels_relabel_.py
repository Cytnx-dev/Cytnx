uT = cytnx.UniTensor.arange(2*3*4).reshape(2,3,4) \
                    .set_rowrank(1).set_name("uT")

uT.relabel_(1,"xx")
uT.print_diagram()

uT.relabel_(["a","b","c"])
uT.print_diagram()
