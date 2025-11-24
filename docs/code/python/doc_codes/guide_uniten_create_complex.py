# initialize a tensor with complex data type
T = cytnx.zeros([2,3,4], dtype=cytnx.Type.ComplexDouble)
# convert to UniTensor
uT = cytnx.UniTensor(T, name="zeros")
# randomize the elements with a uniform distribution in the range [low, high]
cytnx.random.uniform_(uT, low = -1., high = 1.)
# visualize UniTensor
uT.print_diagram()