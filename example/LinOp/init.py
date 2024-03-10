import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
from cytnx import *


# LinOp class provides a base class that defines the operation on a vector.
# This class or it's derived class are required
# for using the cytnx's iterative solver such as Lanczos and Arnodi.

#-----------------------------------------
# Suppose we want to define a custom linear operation
# acting on input vector t (dim=4) that swap the first and last element
# and add the 2nd and 3rd element with one.

t = arange(4)
print(t)

# Method 1, write a custom function, and assign into the LinOp class
# -----------------------------------
# [Note] the function should have the signature
#        Tensor f(const Tensor&) as in C++
def myfunc(v):
    out = v.clone();
    out[0], out[3] = v[3], v[0]; #swap
    out[1]+=1 #add 1
    out[2]+=1 #add 1
    return out;


## at init, we need to specify
## the dtype and device of the input and output vectors of custom function "myfunc".
## Here, it's double type and on cpu.

# Do not use LinOp constructor directly in python, the custom_f argument is not supported.
# Instead, create your own class that inherit LinOp class, and overload the matvec() function.

# lop = LinOp("mv",nx=4,dtype=Type.Double,device=Device.cpu,custom_f=myfunc)
# print(lop.matvec(t)) ## use .matvec(t) to get the output.


# Method 2, write a custom class that inherit LinOp class.
# ------------------------------------
# [Note] Instead of writing a custom class, we overload the matvec() directly.
#        Inheritance is handy if there are additional parameters/arguments
#        needed for the custom operation.
class MyOp(LinOp):
    AddConst = 1 # let's make it a class member.

    def __init__(self,typ,nx,aconst):
        ## here, we can simply set the type and device explicitly as we overload the matvec.
        LinOp.__init__(self,typ,nx,Type.Double,Device.cpu) ## Remember to init the mother class if you want to overload __init__ function.
        self.AddConst = aconst

    def matvec(self,v):
        out = v.clone()
        out[0],out[3] = v[3],v[0]
        out[1]+=self.AddConst #add the constant
        out[2]+=self.AddConst #add the constant
        return out

mylop = MyOp("mv",nx=4,aconst=3) # let's add 3 instead of one.
print(mylop.matvec(t)) ## use .matvec(t) to get the output as usual.
