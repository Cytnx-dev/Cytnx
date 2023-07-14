Scalar
==========
[v0.7+][C++ only]

Scalar is a generic data type which can hold various data types in C++.  

In Python, the data type of any variable is dynamic, i.e. it can be changed at any time. In contrast to this, C++ data types are static and cannot be changed once they are declared. In the C++/Python cross-platform library Cytnx, we provide a generic data type Scalar on the C++ side which allows variables to be used like dynamic data types in Python, with all the convenience and advantages related to this. 


Define/Declare a Scalar
*************************
Defining a Scalar is straight forward. One can convert directly from a supported standard C++ Type to a Scalar. 

Currently, the following standard C++ data types are supported by Cytnx Scalar:

.. tabularcolumns:: |l|l|l|

+------------------+----------------------+-------------------+
| Cytnx type       | C++ type             | Type object       |
+==================+======================+===================+
| cytnx_double     | double               | Type.Double       |
+------------------+----------------------+-------------------+
| cytnx_float      | float                | Type.Float        |
+------------------+----------------------+-------------------+
| cytnx_uint64     | uint64_t             | Type.Uint64       |
+------------------+----------------------+-------------------+
| cytnx_uint32     | uint32_t             | Type.Uint32       |
+------------------+----------------------+-------------------+
| cytnx_uint16     | uint16_t             | Type.Uint16       |
+------------------+----------------------+-------------------+
| cytnx_int64      | int64_t              | Type.Int64        |
+------------------+----------------------+-------------------+
| cytnx_int32      | int32_t              | Type.Int32        |
+------------------+----------------------+-------------------+
| cytnx_int16      | int16_t              | Type.Int16        |
+------------------+----------------------+-------------------+
| cytnx_complex128 | std::complex<double> | Type.ComplexDouble|
+------------------+----------------------+-------------------+
| cytnx_complex64  | std::complex<float>  | Type.ComplexFloat |
+------------------+----------------------+-------------------+
| cytnx_bool       | bool                 | Type.Bool         |
+------------------+----------------------+-------------------+

Consider the example where we want to create a Scalar with *double* type. The conversion between C++ standard type to Scalar is very flexible, depending on the application scenario:

* 1. Convert from a C++ variable

.. literalinclude:: ../../../code/cplusplus/guide_codes/5_1_ex1.cpp
    :language: c++
    :linenos:

Output:

.. literalinclude:: ../../../code/cplusplus/outputs/5_1_ex1.out
    :language: text

* 2. Create directly:

.. literalinclude:: ../../../code/cplusplus/guide_codes/5_1_ex2.cpp
    :language: c++
    :linenos:


Output:

.. literalinclude:: ../../../code/cplusplus/outputs/5_1_ex2.out
    :language: text


.. Note::
    
    1. We can also just assign a C++ type to Scalar as in the case of **A2**. 

    2. In the case of variable **A3**, the assigned value *10* by default in C++ is an integer type. If we want the Scalar to be of type *double*, we can pass an additional dtype to further specify the type explicitly. 

* 3. Convert to a C++ data type:

    Converting to a C++ data type works just like other type castings in C++:

.. literalinclude:: ../../../code/cplusplus/guide_codes/5_1_ex3.cpp
    :language: c++
    :linenos:


Output:

.. literalinclude:: ../../../code/cplusplus/outputs/5_1_ex3.out
    :language: text


.. note::

    Note the slightly different syntax in case of complex type castings. Use **complex128()** and **complex64()** to convert to a standard C++ type **complex<double>** and **complex<float>** respectively.


Change data type
******************
    To change the data type of a Scalar, use **.astype()**. 

.. literalinclude:: ../../../code/cplusplus/guide_codes/5_2_ex1.cpp
    :language: c++
    :linenos:

Output:

.. literalinclude:: ../../../code/cplusplus/outputs/5_2_ex1.out
    :language: text


Application scenarios
**********************
The Scalar type allows for many possibilities in C++ that are otherwise hard to achieve. We will consider some examples in the following.

* 1. If we want to have a list (vector) with elements having different data types, we can use *tuple* objects in C++. However, that requires the number of elements and data types to be known and fixed a priori. Using Scalar, we can also create a vector with a variable number of elements and variable data types:

.. literalinclude:: ../../../code/cplusplus/guide_codes/5_3_ex1.cpp
    :language: c++
    :linenos:

Output:

.. literalinclude:: ../../../code/cplusplus/outputs/5_3_ex1.out
    :language: text


* 2. In C++, unlike Python, we can not create a function that can take arbitrary types of arguments. This becomes possible with Scalar:

.. code-block:: c++
    :linenos:

    Scalar generic_func(const std::vector<Scalar> &args){
        // do something here with args

    }


This way, the user can dynamically decide which types of variables to pass to the function. 


.. toctree::
    :maxdepth: 1

