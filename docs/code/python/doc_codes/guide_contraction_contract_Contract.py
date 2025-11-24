A = cytnx.UniTensor.ones([2,3,4],
                    rowrank=1,
                    labels=["i","j","l"])
Are = A.relabels(["i","j","lA"])

B = cytnx.UniTensor.ones([3,2,4,5],
                    rowrank=2,
                    labels=["j","k","l","m"])
Bre = B.relabels(["j","k","lB","m"])

C = cytnx.Contract(Are, Bre)

A.print_diagram()
B.print_diagram()
C.print_diagram()
