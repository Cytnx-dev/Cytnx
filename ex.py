import os,sys
import cytnx as cy
from cytnx import cytnx_extension as cyx

A = cy.arange(200,dtype=cy.Double).reshape(4,5,5,2)
TA = cyx.CyTensor(A,2)
TA.set_labels([-1,-3,-5,-6])
TA.print_diagram()

TA.permute_([-3,-5,-6,-1],by_label=True)
TA.print_diagram()

