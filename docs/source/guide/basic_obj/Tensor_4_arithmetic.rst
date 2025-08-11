Tensor arithmetic
----------------------

In Cytnx, arithmetic operations such as **+, -, *, /, +=, -=, *=, /=** can be performed between a Tensor and either another Tensor or a scalar, just like the standard way it is done in Python. See also :ref:`Linear algebra` for a list of arithmetic operations and further linear algebra functions.

Type promotion
********************
Arithmetic operations in Cytnx follow a similar pattern of type promotion as standard C++/Python.
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
Several arithmetic operations between a Tensor and a scalar are possible.
For example:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Tensor_4_arithmetic_tensor_scalar.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Tensor_4_arithmetic_tensor_scalar.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Tensor_4_arithmetic_tensor_scalar.out
    :language: text


Tensor-Tensor arithmetic
****************************
Arithmetic operations between two Tensors of the same shape are possible.
For example:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Tensor_4_arithmetic_tensor_tensor.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Tensor_4_arithmetic_tensor_tensor.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Tensor_4_arithmetic_tensor_tensor.out
    :language: text

.. Note::

    An elementwise multiplication is applied if the operator **'*'** is used. For tensor contractions or matrix multiplications, see :ref:`Contraction`.


Equivalent APIs (C++ only)
****************************
Cytnx also provides some equivalent APIs for users who are familiar with/coming from pytorch and similar libraries.
For example, there are two different ways to perform the + operation: **Tensor.Add()/Tensor.Add_()** and **linalg.Add()**

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Tensor_4_arithmetic_Add.cpp
    :language: c++
    :linenos:

.. Note::

    1. All the arithmetic functions such as **Add,Sub,Mul,Div...**, as well as the linear algebra functions all start with capital characters. Beware, since they all start with lower-case characters in pytorch.
    2. All the arithmetic operations with an underscore (such as **Add_, Sub_, Mul_, Div_**) are inplace versions that modify the current instance.

.. Hint::

    1. If the input is of type ComplexDouble/ComplexFloat/Double/Float and both inputs are of the same type, the arithmetic operations internally call BLAS/cuBLAS/MKL `?axpy <https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2023-0/axpy.html>`_ routines.
    2. Arithmetic operations between other types (including different types) are accelerated with OpenMP on the CPU. On a GPU, custom kernels are used to perform the operations.


.. toctree::
