T = cytnx.UniTensor(cytnx.ones([5,5,5]), rowrank = 2, labels = ["a","b","c"])
T.print_diagram()
print("Rowrank of T = ", T.rowrank())
T.Transpose().print_diagram()  # print the transposed T
print("Rowrank of transposed T = ", T.Transpose().rowrank())
