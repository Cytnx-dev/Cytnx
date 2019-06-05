from cytnx import *

A = Tensor([3,4,5])

A.to_(Device.cuda+0);
print(A.device_str())

