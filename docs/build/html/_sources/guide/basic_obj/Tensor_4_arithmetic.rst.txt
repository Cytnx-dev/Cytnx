Tensor arithmetic
----------------------

In cytnx, Tensor can performs arithmetic operation such as **+, -, x, /, +=, -=, *=, /=** with another Tensor or scalalr, just like the standard way you do in python. 

Type promotion
********************
Arithmetic operation in Cytnx follows the similar pattern of type promotion as standard C++/python. 
When Tensor performs arithmetic operation with another Tensor or scalar, the output Tensor will have the dtype as the one that has stronger type. 

The Types order from strong to weak as:
 
    * Type.ComplexDouble 
    * Type.ComplexFloat 
    * Type.Double
    * Type.Float
    * Type.Int64
    * Type.Uint64
    * Type.Int32
    * Type.Uint32
    * Type.Int16
    * Type.Uint16
    * Type.Bool 


Tensor-Tensor arithmetic
****************************
Tensor can performs arithmetic operation with another Tensor with the same shape. 


Tensor-scalar arithmetic
*****************************
Tensor can also performs arithmetic operation with scalar. 



Equivalent APIs
*********************
Following are some equivalent APIs that  are also provided in Cytnx for users who are familiar and coming from pytorch and other librariy communities. 



.. Note::

    1. All the arithmetic operation function such as **Add,Sub,Mul,Div...**, as well as linear algebra functions all start with capital characters. 
    While in pytorch, they are all lower-case. 
    2. All the arithmetic operations with a underscore (such as **Add_, Sub_, Mul_, Div_**)are the inplace version that modify the current instance. 

.. Hint::
    
    1. ComplexDouble/ComplexFloat/Double/Float, these 4 types internally calls BLAS/cuBLAS/MKL ?axpy when the inputs are in the same types. 
    2. Arithmetic between other types (Including different types) are accelerated with OpenMP on CPU. For GPU, custom kernels are used to perform operation. 


.. toctree::
    :numbered:
