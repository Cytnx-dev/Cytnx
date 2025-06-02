bd1 = cytnx.Bond(cytnx.BD_IN,[[1],[-1]],[1,1])
bd2 = cytnx.Bond(cytnx.BD_IN,[[1],[-1]],[1,1])
bd3 = cytnx.Bond(cytnx.BD_OUT,[[2],[0],[0],[-2]],[1,1,1,1])

T = cytnx.UniTensor([bd1,bd2,bd3], rowrank = 2, labels = ["a","b","c"])
T.print_diagram()
print("Rowrank of T = ", T.rowrank())
T.Transpose().print_diagram()  # print the transposed T
print("Rowrank of transposed T = ", T.Transpose().rowrank())
