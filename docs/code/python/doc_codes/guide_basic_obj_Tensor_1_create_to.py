A = cytnx.ones([2,2]) #on CPU
B = A.to(cytnx.Device.cuda+0)
print(A) # on CPU
print(B) # on GPU

A.to_(cytnx.Device.cuda)
print(A) # on GPU
