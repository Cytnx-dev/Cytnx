import sys
sys.path.append("../../Cytnx")
import cytnx
import numpy as np 

A = cytnx.arange(10).reshape(2,5);
B = A.storage();

print(A)
print(B)

print("-----------------")
A = cytnx.arange(8).reshape(2,2,2)
print(A.storage()) 

# Let's make it non-contiguous 
A.permute_(0,2,1)
print(A.is_contiguous()) 

# Note that the storage is not changed
print(A.storage())

# Now let's make it contiguous
# thus the elements is moved
A.contiguous_();
print(A.is_contiguous())

# Note that the storage now is changed 
print(A.storage())

