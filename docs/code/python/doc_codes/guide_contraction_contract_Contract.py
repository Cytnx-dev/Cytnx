A = cytnx.UniTensor.ones([2,3,4]) \
                   .set_rowrank(1) \
                   .relabel(["i","j","l"]) \
                   .set_name("A")
Are = A.relabel(["i","j","lA"]).set_name("Are")

B = cytnx.UniTensor.ones([3,2,4,5]) \
                   .set_rowrank(2) \
                   .relabel(["j","k","l","m"]) \
                   .set_name("B")
Bre = B.relabel(["j","k","lB","m"]).set_name("Bre")

C = cytnx.Contract(Are, Bre)

A.print_diagram()
B.print_diagram()
C.print_diagram()
