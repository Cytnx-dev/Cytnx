A = cytnx.UniTensor(cytnx.ones([2,3,4]), rowrank = 1)
A.relabels_(["i","j","l"])
Are = A.relabels(["i","j","lA"])

B = cytnx.UniTensor(cytnx.ones([3,2,4,5]), rowrank = 2)
B.relabels_(["j","k","l","m"])
Bre = B.relabels(["j","k","lB","m"])

C = cytnx.Contract(Are, Bre)

A.print_diagram()
B.print_diagram()
C.print_diagram()
