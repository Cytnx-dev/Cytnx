T = cytnx.UniTensor.ones([5,5,5]).relabel(["a","b","c"]) \
    .set_rowrank(2).set_name("T")
T.print_diagram()
print("Rowrank of T = ", T.rowrank())
T.Transpose().print_diagram()  # print the transposed T
print("Rowrank of transposed T = ", T.Transpose().rowrank())
