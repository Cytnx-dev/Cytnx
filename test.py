import numpy as np
import cytnx as cy
from cytnx import cytnx_extension as cyx

a = cy.from_numpy(np.arange(1.0, 3.0))
Ta = cyx.CyTensor(a, 1)
Ta.set_name('Ta')
Ta.print_diagram()
print(Ta)

b = cy.from_numpy(np.arange(3.0, 5.0))
Tb = cyx.CyTensor(b, 0)
Tb.set_labels([1])
Tb.set_name('Tb')
Tb.print_diagram()
print(Tb)

Tab = cyx.Contract(Ta, Tb)
Tab.set_name('Tab')
Tab.print_diagram()
print(Tab)
print(Tab.get_block())

Tba = cyx.Contract(Tb, Ta)
Tba.set_name('Tba')
Tba.print_diagram()
print(Tba)
print(Tba.get_block())






