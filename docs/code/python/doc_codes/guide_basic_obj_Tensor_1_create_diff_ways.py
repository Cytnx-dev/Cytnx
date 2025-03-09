#rank-1 Tensor from [0,10) with step 1
A = cytnx.arange(10)

#rank-1 Tensor from [0,10) with step 2
B = cytnx.arange(0,10,2)

#Tensor of shape (3,4,5) with all elements 
#set to one.
C = cytnx.ones([3,4,5])

#Tensor of shape (3,3) with diagonal elements 
#set to one, all other entries are zero.
D = cytnx.eye(3)
