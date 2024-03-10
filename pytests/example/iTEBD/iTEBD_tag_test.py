import sys,os
import numpy as np
sys.path.append('example/iTEBD/')

from iTEBD_tag import *

def test_itebd():
    E = itebd_tfim_tag()
    assert np.abs(E-(-2.07854514)) < 1e-8
