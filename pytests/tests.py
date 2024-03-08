import sys, os
sys.path.append('/home/runner/work/Cytnx_lib')

from cytnx import *
T = zeros([4,4])
CyT = UniTensor(T,rowrank=2) #create un-tagged UniTensor from Tensor
CyT.print_diagram()
