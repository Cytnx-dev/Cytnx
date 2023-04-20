Tensor arithmetic
----------------------

In cytnx, arithmetic operations such as **+, -, x, /, +=, -=, *=, /=** can be performed between a Tensor and either another Tensor or a scalar, just like the standard way it is done in python. 

Type promotion
********************
Arithmetic operations in Cytnx follow a similar pattern of type promotion as standard C++/python. 
When an arithmetic operation between a Tensor and another Tensor or scalar is performed, the output Tensor will have the same dtype as the input with the stronger type. 

The Type order from strong to weak is:
 
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
Arithmetic operations between a Tensor and a scalar can be performed. 
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
Arithmetic operations between two Tensors of the same shape are possible. 
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
Cytnx also provides some equivalent APIs for users who are familiar with/coming from pytorch and similar libraries. 
For example, there are two different ways to perform the + operation: **Tensor.Add()/Tensor.Add_()** and **linalg.Add()**

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

    1. All the arithmetic operation functions such as **Add,Sub,Mul,Div...**, as well as the linear algebra functions all start with capital characters. Beware, since they all start with lower-case characters in pytorch.
    2. All the arithmetic operations with an underscore (such as **Add_, Sub_, Mul_, Div_**) are inplace versions that modify the current instance. 

.. Hint::
    
    1. If the input is of type ComplexDouble/ComplexFloat/Double/Float and both inputs are of the same type, the arithmetic operations internally call BLAS/cuBLAS/MKL ?axpy. 
    2. Arithmetic operations between other types (including different types) are accelerated with OpenMP on the CPU. On a GPU, custom kernels are used to perform the operations. 


.. toctree::
