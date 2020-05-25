from cytnx import *
from cytnx import cytnx_extension as cyx


T = zeros([4,4])
CyT = cyx.CyTensor(T,2) #create un-tagged CyTensor from Tensor
CyT.print_diagram()

print("before:")
print(T)
print(CyT)

#Note that it is a shared view, so a change to CyT will affect Tensor T.
CyT.set_elem([0,0],1.456)

print("after")
print(T)
print(CyT)

#If we want a new instance of memery, use clone at initialize:
print("[non-share example]")
CyT_nonshare = cyx.CyTensor(T.clone(),2);

print("before:")
print(T)
print(CyT_nonshare)

CyT_nonshare.set_elem([1,1],2.345);
    
print("after")
print(T) # T is unchanged!
print(CyT_nonshare)





