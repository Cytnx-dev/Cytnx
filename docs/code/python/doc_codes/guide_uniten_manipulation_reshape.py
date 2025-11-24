T = cytnx.UniTensor.arange(12).reshape(4,3) \
         .relabel(["a", "b"]).set_name("T")
T.reshape_(2,3,2)
T.print_diagram()
