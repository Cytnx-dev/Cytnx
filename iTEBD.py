import numpy as np
import scipy as sp
from scipy import linalg
import sys
sys.path.append("cytnx")
import cytnx
from cytnx import cytnx_extension as cyx

## Example of 1D Ising model 
## iTEBD
##-------------------------------------

chi = 20
Hx = 1.0
CvgCrit = 1.0e-10
dt = 0.1

## Create onsite-Op
Sz = cytnx.zeros([2,2])
Sz[0,0] = 1
Sz[1,1] = -1

Sx = cytnx.zeros([2,2])
Sx[0,1] = Sx[1,0] = Hx

I = Sz.clone()
I[1,1] = 1 

#print(Sz,Sx)

## Build Evolution Operator
TFterm = cytnx.linalg.Kron(Sx,I) + cytnx.linalg.Kron(I,Sx)
ZZterm = cytnx.linalg.Kron(Sz,Sz)

H = TFterm + ZZterm 
del TFterm, ZZterm

eH = cytnx.linalg.ExpH(H,-dt) ## or equivantly ExpH(-dt*H)
eH.reshape_(2,2,2,2)
print(eH)

eH = cyx.CyTensor(eH,2)
eH.print_diagram()
print(eH)


## Create MPS:
#
#     |    |     
#   --A-la-B-lb-- 
#
A = cyx.CyTensor([cyx.Bond(chi),cyx.Bond(2),cyx.Bond(chi)],rowrank=1,labels=[-1,0,-2])
B = cyx.CyTensor(A.bonds(),rowrank=1,labels=[-3,1,-4])
A.print_diagram()
B.print_diagram()
la = cyx.CyTensor([cyx.Bond(chi),cyx.Bond(chi)],rowrank=1,labels=[-2,-3],is_diag=True)
lb = cyx.CyTensor([cyx.Bond(chi),cyx.Bond(chi)],rowrank=1,labels=[-4,-5],is_diag=True)
la.print_diagram()
lb.print_diagram()



## Evov:
Elast = 0
for i in range(10000):
    A.set_labels([-1,0,-2])
    B.set_labels([-3,1,-4])
    la.set_labels([-2,-3])
    lb.set_labels([-4,-5])

    X = cyx.Contract(cyx.Contract(A,la),cyx.Contract(B,lb))
    X.print_diagram()
    exit(1)


