import sys
sys.path.append("../../Cytnx")
import cytnx
import numpy as np 

def myfunc(v):
    out = v.clone()
    out[0],out[3] = v[3], v[0] #swap
    out[1]+=1 #add 1
    out[2]+=1 #add 1
    return out


H = cytnx.LinOp("mv",nx=4,\
                dtype=cytnx.Type.Double,\
                device=cytnx.Device.cpu,\
                custom_f=myfunc)


x = cytnx.arange(4)
y = H.matvec(x)
print(x)
print(y)

