import sys
sys.path.append("../../Cytnx")
import cytnx
import numpy as np 

class MyOp(cytnx.LinOp):
    AddConst = 1# class member.

    def __init__(self,aconst):
        # here, we fix nx=4, dtype=double on CPU, 
        # so the constructor only take external argument 'aconst'

        ## Remember to init the mother class. 
        ## Here, we don't specify the custom_f!
        cytnx.LinOp.__init__(self,"mv",4,cytnx.Type.Double,\
                                   cytnx.Device.cpu )

        self.AddConst = aconst

    def matvec(self, v):
        out = v.clone()
        out[0],out[3] = v[3],v[0] # swap
        out[1]+=self.AddConst #add constant
        out[2]+=self.AddConst #add constant
        return out

myop = MyOp(7)
x = cytnx.arange(4)
y = myop.matvec(x)

print(x)
print(y)

