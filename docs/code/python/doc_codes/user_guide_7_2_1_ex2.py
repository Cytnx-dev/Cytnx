# initialize a tensor with complex data type
T = cytnx.zeros([2,3,4], dtype=cytnx.Type.ComplexDouble)
# convert to UniTensor
uT = cytnx.UniTensor(T)
# randomize the elements with a uniform distribution in the range [low, high]
cytnx.random.Make_uniform(uT, low = -1., high = 1.)
