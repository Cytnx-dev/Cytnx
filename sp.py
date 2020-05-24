import cytnx as cy
from cytnx import cytnx_extension as cyx


## spin-1 example
bi = cyx.Bond(3,cyx.BD_KET,[[2],[0],[-2]],[cyx.Symmetry.U1()])
bo = cyx.Bond(3,cyx.BD_BRA,[[2],[0],[-2]],[cyx.Symmetry.U1()])

A = cyx.CyTensor([bi,bi,bo,bo],rowrank=2)
A.print_diagram()
B = A.clone()

Heisenberg = cy.linalg.Kron(cy.physics.spin(1,'z'),cy.physics.spin(1,'z'))\
           + cy.linalg.Kron(cy.physics.spin(1,'y'),cy.physics.spin(1,'y'))\
           + cy.linalg.Kron(cy.physics.spin(1,'x'),cy.physics.spin(1,'x'))
Heisenberg = Heisenberg.real() # it's real so let's truncate imag part.
print(Heisenberg)


Heisenberg.reshape_(3,3,3,3)



## method 1, directly access element, even tho it is sparse storage. 
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                if A.elem_exists([i,j,k,l]):
                    print(i,j,k,l)
                    A.set_elem([i,j,k,l],Heisenberg[i,j,k,l].item())



## method 2, use get_block_() to get reference and put it in. 
Block_q4 = Heisenberg[0,0,0,0].reshape(1,1)

Block_q2 = cy.zeros([2,2])
Block_q2[0,0] = Heisenberg[0,1,0,1]
Block_q2[0,1] = Heisenberg[0,1,1,0]
Block_q2[1,0] = Heisenberg[1,0,0,1]
Block_q2[1,1] = Heisenberg[1,0,1,0]

Block_q0 = cy.zeros([3,3])
Block_q0[0,0] = Heisenberg[0,2,0,2]
Block_q0[0,1] = Heisenberg[0,2,1,1]
Block_q0[0,2] = Heisenberg[0,2,2,0]
Block_q0[1,0] = Heisenberg[1,1,0,2]
Block_q0[1,1] = Heisenberg[1,1,1,1]
Block_q0[1,2] = Heisenberg[1,1,2,0]
Block_q0[2,0] = Heisenberg[2,0,0,2]
Block_q0[2,1] = Heisenberg[2,0,1,1]
Block_q0[2,2] = Heisenberg[2,0,2,0]

Block_qm2 = cy.zeros([2,2])
Block_qm2[0,0] = Heisenberg[1,2,1,2]
Block_qm2[0,1] = Heisenberg[1,2,2,1]
Block_qm2[1,0] = Heisenberg[2,1,1,2]
Block_qm2[1,1] = Heisenberg[2,1,2,1]

Block_qm4 = Heisenberg[2,2,2,2].reshape(1,1)

B.put_block(Block_q4,[4])
B.put_block(Block_q2,[2])
B.put_block(Block_q0,[0])
B.put_block(Block_qm2,[-2])
B.put_block(Block_qm4,[-4])

print(A)
print(B)


