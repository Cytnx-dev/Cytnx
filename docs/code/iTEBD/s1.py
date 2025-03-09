import sys
sys.path.append("/home/kaywu/Dropbox/Cytnx")
import cytnx
import cytnx.cytnx_extension as cyx

chi = 10
A = cyx.CyTensor([cyx.Bond(chi),cyx.Bond(2),cyx.Bond(chi)],rowrank=1,labels=[-1,0,-2])
B = cyx.CyTensor(A.bonds(),rowrank=1,labels=[-3,1,-4])
cytnx.random.Make_normal(B.get_block_(),0,0.2)
cytnx.random.Make_normal(A.get_block_(),0,0.2)
A.print_diagram()
B.print_diagram()

la = cyx.CyTensor([cyx.Bond(chi),cyx.Bond(chi)],rowrank=1,labels=[-2,-3],is_diag=True)
lb = cyx.CyTensor([cyx.Bond(chi),cyx.Bond(chi)],rowrank=1,labels=[-4,-5],is_diag=True)
la.put_block(cytnx.ones(chi))
lb.put_block(cytnx.ones(chi))

la.print_diagram()
lb.print_diagram()

