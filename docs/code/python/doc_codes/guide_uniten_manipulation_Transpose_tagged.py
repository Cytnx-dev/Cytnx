bd1 = cytnx.Bond(cytnx.BD_IN,[[1],[-1]],[1,1])
bd2 = cytnx.Bond(cytnx.BD_IN,[[1],[-1]],[1,1])
bd3 = cytnx.Bond(cytnx.BD_OUT,[[2],[0],[0],[-2]],[1,1,1,1])
uT = cytnx.UniTensor([bd1,bd2,bd3]).set_rowrank(2) \
          .relabel(["a","b","c"]).set_name("uT")
uT.print_diagram()
print("Rowrank of T = ", uT.rowrank())
uTt = uT.Transpose().set_name("uT transposed")
uTt.print_diagram()  # print the transposed uT
print("Rowrank of transposed T = ", uTt.rowrank())
