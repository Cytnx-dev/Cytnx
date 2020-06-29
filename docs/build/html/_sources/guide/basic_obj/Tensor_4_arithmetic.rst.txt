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



Tensor-scalar arithmetic
*****************************
Tensor can also performs arithmetic operation with scalar. 
For example:

* In python:

.. code-block:: python 
    :linenos:        

        A = cytnx.ones([3,4])
        print(A)

        B = A + 4 
        print(B)

        C = A - 7j # type promotion
        print(C)

        

* In C++:

.. code-block:: c++
    :linenos:

        auto A = cytnx::ones({3,4});
        cout << A << endl;

        auto B = A + 4;
        cout << B << endl;

        auto C = A - std::complex<double>(0,7); //type promotion
        cout << C << endl;

Output>>

.. code-block:: text

    Total elem: 12
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (3,4)
    [[1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
     [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
     [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]]


    Total elem: 12
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (3,4)
    [[5.00000e+00 5.00000e+00 5.00000e+00 5.00000e+00 ]
     [5.00000e+00 5.00000e+00 5.00000e+00 5.00000e+00 ]
     [5.00000e+00 5.00000e+00 5.00000e+00 5.00000e+00 ]]


    Total elem: 12
    type  : Complex Double (Complex Float64)
    cytnx device: CPU
    Shape : (3,4)
    [[1.00000e+00-7.00000e+00j 1.00000e+00-7.00000e+00j 1.00000e+00-7.00000e+00j 1.00000e+00-7.00000e+00j ]
     [1.00000e+00-7.00000e+00j 1.00000e+00-7.00000e+00j 1.00000e+00-7.00000e+00j 1.00000e+00-7.00000e+00j ]
     [1.00000e+00-7.00000e+00j 1.00000e+00-7.00000e+00j 1.00000e+00-7.00000e+00j 1.00000e+00-7.00000e+00j ]]



Tensor-Tensor arithmetic
****************************
Tensor can performs arithmetic operation with another Tensor with the same shape. 
For example:

* In python:

.. code-block:: python 
    :linenos:        

        A = cytnx.arange(12).reshape(3,4)
        print(A)

        B = cytnx.ones([3,4])*4 
        print(B)

        C = A * B
        print(C)

        

* In C++:

.. code-block:: c++
    :linenos:

        auto A = cytnx::arange(12).reshape(3,4);
        cout << A << endl;

        auto B = cytnx.ones({3,4})*4;
        cout << B << endl;

        auto C = A * B;
        cout << C << endl;

Output>>

.. code-block:: text

    Total elem: 12
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (3,4)
    [[0.00000e+00 1.00000e+00 2.00000e+00 3.00000e+00 ]
     [4.00000e+00 5.00000e+00 6.00000e+00 7.00000e+00 ]
     [8.00000e+00 9.00000e+00 1.00000e+01 1.10000e+01 ]]


    Total elem: 12
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (3,4)
    [[4.00000e+00 4.00000e+00 4.00000e+00 4.00000e+00 ]
     [4.00000e+00 4.00000e+00 4.00000e+00 4.00000e+00 ]
     [4.00000e+00 4.00000e+00 4.00000e+00 4.00000e+00 ]]


    Total elem: 12
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (3,4)
    [[0.00000e+00 4.00000e+00 8.00000e+00 1.20000e+01 ]
     [1.60000e+01 2.00000e+01 2.40000e+01 2.80000e+01 ]
     [3.20000e+01 3.60000e+01 4.00000e+01 4.40000e+01 ]]


Equivalent APIs (C++ only)
****************************
Cytnx also provides some equivelant APIs for users who are familiar and coming from pytorch and other library communities. 
For example, suppose we want to do + operation, there are two other ways: Use **Tensor.Add()/Tensor.Add_()** and **linalg.Add()**

* In C++:

.. code-block:: c++
    :linenos:

        auto A = cytnx::ones({3,4})
        auto B = cytnx::arange(12).reshape(3,4);
        
        // these two are equivalent to C = A+B;
        auto C = A.Add(B); 
        auto D = cytnx::linalg.Add(A,B);

        // this is equivalent to A+=B;
        A.Add_(B);


.. Note::

    1. All the arithmetic operation function such as **Add,Sub,Mul,Div...**, as well as linear algebra functions all start with capital characters. While in pytorch, they are all lower-case.
    2. All the arithmetic operations with a underscore (such as **Add_, Sub_, Mul_, Div_**)are the inplace version that modify the current instance. 

.. Hint::
    
    1. ComplexDouble/ComplexFloat/Double/Float, these 4 types internally calls BLAS/cuBLAS/MKL ?axpy when the inputs are in the same types. 
    2. Arithmetic between other types (Including different types) are accelerated with OpenMP on CPU. For GPU, custom kernels are used to perform operation. 


.. toctree::
