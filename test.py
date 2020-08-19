#import numpy as np
import cytnx as cy
#from cytnx import cytnx_extension as cyx


#print(cy.__version__)
#print(cy.__blasINTsize__)
#TNs = cy.arange(16).astype(cy.Type.Double).reshape(4,4);
#TNs = cy.arange(16).astype(cy.Type.Double).reshape(4,4);
#print(cy.linalg.Dot(TNs,TNs))
#print(cy.linalg.Eig(TNs))
#print(cy.linalg.Svd(TNs))
A = cy.arange(4)
A /= 3
print(A)
