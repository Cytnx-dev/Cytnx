# Create a rank-3 UniTensor with shape [2,3,4]
# 1) initialized by zeros, using method chaining (preferred)
uT0 = cytnx.UniTensor.zeros([2,3,4]).relabel(["a","b","c"]).set_name("all zeros")

# 2) initialized by ones, using initializer arguments
uT1 = cytnx.UniTensor.ones([2,3,4], ["a","b","c"], name="all ones")

# 3) initialize by normally distributed elements with mean value 0 and standard deviation 1
uTg = cytnx.UniTensor.normal([2,3,4],0,1).relabel(["a","b","c"]).set_name("Gaussian")

# 4) initialize by uniformly distributed elements between 0 and 100
uTu = cytnx.UniTensor.uniform([2,3,4],0,100).relabel(["a","b","c"]).set_name("random uniform")

# Existing UniTensor instances can be randomized in place either through instance
# methods or the cytnx.random module.
uT_inplace_normal = cytnx.UniTensor.zeros([2,3,4], ["a","b","c"], name="normal template")
cytnx.random.normal_(uT_inplace_normal, mean=0., std=1.)
uT_inplace_uniform = cytnx.UniTensor.zeros([2,3,4], ["a","b","c"], name="uniform template")
uT_inplace_uniform.uniform_(low=0., high=100.)

# 5) initialize with subsequent numbers from 0 to 2*3*4 - 1 = 23
uTarr0 = cytnx.UniTensor.arange(2*3*4).reshape(2,3,4).set_rowrank(1) \
                        .relabel(["a","b","c"]).set_name("range 0, 1, ..., 23")

# 6) initialize with subsequent numbers from starting from 5 to 53 with a stepwidth of 2
uTarr5 = cytnx.UniTensor.arange(5, 53, 2).reshape(2,3,4).set_rowrank(1) \
                        .relabel(["a","b","c"]).set_name("range 5, 7, ...., 53")

# Create a 4x4 identity matrix
uTe = cytnx.UniTensor.eye(4, ["left","right"], name="identity")
