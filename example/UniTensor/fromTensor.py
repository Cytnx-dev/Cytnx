from cytnx import *


T = zeros([4,4])
CyT = UniTensor(T,rowrank=2) #create un-tagged UniTensor from Tensor
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
CyT_nonshare = UniTensor(T.clone(),rowrank=2);

print("before:")
print(T)
print(CyT_nonshare)

CyT_nonshare.set_elem([1,1],2.345);
    
print("after")
print(T) # T is unchanged!
print(CyT_nonshare)





