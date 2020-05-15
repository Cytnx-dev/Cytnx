import numpy as np
import cytnx as cy
from cytnx import cytnx_extension as cyx

a = cy.from_numpy(np.arange(1.0, 5.0).reshape(2,2))
Ta = cyx.CyTensor(a, 2)
Ta.set_name('Ta')
Ta.print_diagram()
print(Ta)

b = cy.from_numpy(np.arange(5.0, 9.0).reshape(2,2))
Tb = cyx.CyTensor(b, 0)
Tb.set_labels([2,3])
Tb.set_name('Tb')
Tb.print_diagram()
print(Tb)

Tab = cyx.Contract(Ta, Tb)
Tab.set_name('Tab')
Tab.print_diagram()
print(Tab)
print(Tab.get_block().reshape(4,4))

Tba = cyx.Contract(Tb, Ta)
Tba.set_name('Tba')
Tba.print_diagram()
print(Tba)
print(Tba.get_block().reshape(4,4))






