import numpy as np
import cytnx as cy
from cytnx import cytnx_extension as cyx


x = cy.arange(10).astype(cy.Type.Double)*1.0e-5
print(cy.linalg.Abs(x))
exit(1)


tt = cyx.Bond(3,cyx.BD_BRA,[[2],[0],[1]],[cyx.Symmetry.U1()])
tt.Save("tt")
print(tt)

tt2 = cyx.Bond.Load("tt.cybd")
print(tt2)
exit(1)


tt = cyx.Symmetry.Zn(2)
tt.Save("z2")

tt = cyx.Symmetry.Load("z2.cysym")



tt = cy.ones([3,3],dtype=cy.Type.ComplexDouble)
print(tt.Norm())
tt@=tt
print(tt)

tt.Save("tt")

t2 = cy.Tensor.Load("tt.cytn")
print(t2)

exit(1)

a = cy.from_numpy(np.arange(1.0, 11.0).reshape(2,5))
Ta = cyx.CyTensor(a, 1)
Ta.tag()
Ta.print_diagram()

b = cyx.xlinalg.Svd(Ta)
b[0].set_name("s")
b[1].set_name("u")
b[2].set_name("vt")

b[0].print_diagram()
b[1].print_diagram()
b[2].print_diagram()


