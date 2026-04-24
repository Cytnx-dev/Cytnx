# initialize a UniTensor with complex data type
uT = cytnx.UniTensor.zeros([2,3,4], dtype=cytnx.Type.ComplexDouble) \
          .set_name("uT").relabel(["a", "b", "c"])
# randomize the elements with a uniform distribution in the range [low, high]
cytnx.random.uniform_(uT, low = -1., high = 1.)
# visualize UniTensor
uT.print_diagram()
