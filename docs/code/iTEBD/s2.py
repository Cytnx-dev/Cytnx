import sys
sys.path.append("/home/kaywu/Dropbox/Cytnx")
import cytnx
import cytnx.cytnx_extension as cyx

J = -1.0
Hx = 0.3
dt = 0.1

## Create single site operator
Sz = cytnx.physics.pauli('z').real()
Sx = cytnx.physics.pauli('x').real()
I  = cytnx.eye(2)
print(Sz)
print(Sx)


## Construct the local Hamiltonian
TFterm = cytnx.linalg.Kron(Sx,I) + cytnx.linalg.Kron(I,Sx)
ZZterm = cytnx.linalg.Kron(Sz,Sz)
H = Hx*TFterm + J*ZZterm 
print(H)


## Build Evolution Operator
eH = cytnx.linalg.ExpH(H,-dt) ## or equivantly ExpH(-dt*H)
eH.reshape_(2,2,2,2)
eH = cyx.CyTensor(eH,2)
eH.print_diagram()


