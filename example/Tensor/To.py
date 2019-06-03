from cytnx import *

A = Tensor([3,4,5])
B = A.to(Device.cuda+0);
print(B.device_str())
print(A.device_str())

A.to_(Device.cuda+0);
print(A.device_str())

