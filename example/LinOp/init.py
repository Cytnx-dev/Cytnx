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


lop = LinOp("mv",myfunc)
print(lop.matvec(t)) ## use .matvec(t) to get the output.


# Method 2, write a custom class that inherit LinOp class.
# ------------------------------------
# [Note] Instead of writing a custom class, we overload the matvec() directly.
#        Inheritance is handy if there are additional parameters/arguments 
#        needed for the custom operation.
class MyOp(LinOp):
    AddConst = 1 # let's make it a class member.

    def __init__(self,typ,aconst):
        LinOp.__init__(self,typ) ## Remember to init the mother class if you want to overload __init__ function.
        self.AddConst = aconst

    def matvec(self,v):
        out = v.clone()
        out[0],out[3] = v[3],v[0]
        out[1]+=self.AddConst #add the constant
        out[2]+=self.AddConst #add the constant
        return out

mylop = MyOp("mv",3) # let's add 3 instead of one. 
print(mylop.matvec(t)) ## use .matvec(t) to get the output as usual.








