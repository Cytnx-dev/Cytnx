import cytnx
import cytnx.cytnx_extension as cyx


J  = 1.0
hx = 0.2

## Spin 1/2 operator
Sz = cytnx.zeros([2,2])
Sz[0,0] = 1;
Sz[1,1] = -1;

Sx = cytnx.zeros([2,2])
Sx[0,1] = Sx[1,0] = 1;

I  = cytnx.linalg.Diag(cytnx.ones(2))

## construct MPO
MPO = cytnx.zeros([3,3,2,2])
MPO[0,0,:,:] = I
MPO[1,0,:,:] = J*Sz
MPO[2,0,:,:] = -hx*Sx
MPO[2,1,:,:] = J*Sz
MPO[2,2,:,:] = I


## as CyTensor :
MPO_T = cyx.CyTensor(MPO,2)
MPO_T.print_diagram()








