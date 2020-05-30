import numpy as np
import cytnx as cy
from cytnx import cytnx_extension as cyx


tt = cy.ones([3,3],dtype=cy.Type.ComplexDouble)
print(tt.Norm())
tt@=tt
print(tt@tt)

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


