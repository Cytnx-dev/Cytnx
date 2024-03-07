from cytnx import *

# pytest test

T = zeros([4,4])
CyT = UniTensor(T,rowrank=2) #create un-tagged UniTensor from Tensor
CyT.print_diagram()
