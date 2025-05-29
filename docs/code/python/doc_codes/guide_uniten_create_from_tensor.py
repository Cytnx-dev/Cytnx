# create a rank-3 tensor with shape [2,3,4]
T = cytnx.arange(2*3*4).reshape(2,3,4)
# convert to UniTensor:
uT = cytnx.UniTensor(T)
