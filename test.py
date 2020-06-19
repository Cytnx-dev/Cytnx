#import numpy as np
import cytnx as cy
from cytnx import cytnx_extension as cyx


print(cy.__version__)
print(cy.__blasINTsize__)
TNs = cy.arange(16).astype(cy.Type.Double).reshape(4,4);
TNs = cy.arange(16).astype(cy.Type.Double).reshape(4,4);
print(cy.linalg.Dot(TNs,TNs))
#print(cy.linalg.Eig(TNs))
print(cy.linalg.Svd(TNs))

exit(1);
a = cy.from_numpy(np.arange(1.0, 11.0).reshape(2,5))

cy.linalg.Svd(a)

exit(1)
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


