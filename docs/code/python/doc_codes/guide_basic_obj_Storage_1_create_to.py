A = cytnx.Storage(4)
B = A.to(cytnx.Device.cuda)

print(A.device_str())
print(B.device_str())

A.to_(cytnx.Device.cuda)
print(A.device_str())
