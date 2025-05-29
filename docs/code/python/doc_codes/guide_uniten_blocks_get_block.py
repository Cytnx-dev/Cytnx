# Create an UniTensor from Tensor
T = cytnx.UniTensor(cytnx.ones([3,3]))
print(T.get_block())
