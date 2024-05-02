import sys,os
import numpy as np
sys.path.append('example/ED/')

from ed_ising import *

def test_ed_ising():
    L = 4
    J = 1
    Hx = 0.3
    H = Hising(L,J,Hx)
    v = cy.ones(16)
    res = cy.linalg.Lanczos(Hop = H, Tin = v, method = "ER", max_krydim = 2)
    assert np.abs(res[0].item()-(-4.092961599426854)) < 1e-4
