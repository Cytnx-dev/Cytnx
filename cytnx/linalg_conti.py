from .utils import *
from cytnx import *


def Lanczos_ER(Hop,k=1,is_V=True,maxiter=10000, CvgCrit=1.0e-14,is_row=False,Tin=Tensor(),max_krydim=4, verbose=False):
    #print(verbose)
    return linalg.c_Lanczos_ER(Hop,k,is_V,maxiter,CvgCrit,is_row,Tin,max_krydim,verbose);


# inject into the submodule
obj = cytnx.linalg
setattr(obj,"Lanczos_ER",Lanczos_ER)

