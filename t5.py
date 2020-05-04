import cytnx
import numpy 
import cytnx.cytnx_extension as cyx

A = cytnx.arange(40).reshape(5,4,2)


cA = cyx.CyTensor(A,1)
cB = cyx.CyTensor(A,1)
cC = cyx.CyTensor(A,1)
cD = cyx.CyTensor(A,1)
N = cyx.Network("t.net")
N.Diagram()

