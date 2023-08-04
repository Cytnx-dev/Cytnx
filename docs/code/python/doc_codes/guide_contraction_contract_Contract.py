A = cytnx.UniTensor(cytnx.ones([2,3,4]), rowrank = 1)
A.relabels_(["i","j","l"])

B = cytnx.UniTensor(cytnx.ones([3,2,4,5]), rowrank = 2)
B.relabels_(["j","k","l","m"])

C = cytnx.Contract(A, B)

A.print_diagram()
B.print_diagram()
C.print_diagram()
