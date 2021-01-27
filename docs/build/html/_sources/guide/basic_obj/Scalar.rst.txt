Scalar
==========
[v0.7+][C++ only]

Scalar is the generic data type to hold all types in C++.  

In python, the variable's type is dynamic, i.e. they can be changed at any time. Compare to C++, where the data type of a variable are static, and once declared, they cannot be changed. In Cytnx, as a library across C++/python, we provide a generic type Scalar which allow the variable to take the convenience and advantage of the dynamic type as python. 


Define/Declare a scalar
*************************
To define a scalar is pretty simple. One can convert directly from a supported standard C++ Type. 

Currently, following standard C++ data types are supported by cytnx Scalar:

.. tabularcolumns:: |l|l|l|

+------------------+----------------------+-------------------+
| cytnx type       | c++ type             | Type object       |
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

* 1. Convert from a c++ variable

.. code-block:: c++
    :linenos:

    double cA = 1.33;
    Scalar A(cA);
    cout << A << endl;

Output:
 
.. code-block:: text

    < 1.33 >  Scalar dtype: [Double (Float64)]


* 2. Create directly:

.. code-block:: c++
    :linenos:

    Scalar A(double(1.33));

    Scalar A2 = double(1.33);

    Scalar A3(10,Type.Double);

    cout << A << A2 << A3 << endl;

Output:

.. code-block:: text

    < 1.33 >  Scalar dtype: [Double (Float64)]
    < 1.33 >  Scalar dtype: [Double (Float64)]
    < 10 >  Scalar dtype: [Double (Float64)]



.. Note::
    
    1. We can also just assign a c++ type to Scalar as **A2**. 

    2. The variable using c++ template to detect the type, in case of variable **A3**, *10* by defalt in c++ is an integer type. In such case if we want the scalar to be *double*, then one can pass addtional dtype to further specify the type explicitly. 


* 3. Convert to a c++ data type:

    Converting to a c++ data type is very straight. Simply doing exactly the same way as type casting in C++:


.. code-block:: c++
    :linenos:

    Scalar A = 10;
    cout << A << endl;

    auto fA = float(A); // convert to float
    cout << typeid(fA).name() << fA << endl;
    
    // convert to complex double
    auto cdA = complex128(A);
    cout << cdA << endl;

    // convert to complex float
    auto cfA = complex64(A); 
    cout << cfA << endl;

Output:

.. code-block:: text

    < 10 >  Scalar dtype: [Int32]

    f10
    (10,0)
    (10,0)

.. note::

    The case of complex type is a bit different, one use **complex128()** and **complex64()** to convert to a standard c++ type **complex<double>** and **complex<float>** respectively.


Change data type
******************
    To change the data type of Scalar, use **.astype()**. 


.. code-block:: c++
    :linenos:

    Scalar A(1.33);
    cout << A << endl;

    A = A.astype(Type.Float);
    cout << A << endl;

Output:

.. code-block:: text
    
    < 1.33 >  Scalar dtype: [Double (Float64)]

    < 1.33 >  Scalar dtype: [Float (Float32)]



Application scenarios
**********************
With the Scalar type, there are much more possibilities in c++. In the following, let's consider some examples

* 1. Consider the scenario where we want to have a list (vector) with each elements to be different data types. Indeed, in C++ we can use *tuple*. However, that requires the number of elements and data types to be known and fixed a priori. If, say, we want to return a vector with known number of elements, and known data types. Using Scalar we can do:

.. code-block:: c++
    :linenos:

    vector<Scalar> out;

    out.push_back(Scalar(1.33)); //double
    out.push_back(Scalar(10));   //int
    out.push_back(Scalar(cytnx_complex128(3,4))); //complex double

    cout << out[0] << out[1] << out[2] << endl;

Output:

.. code-block:: text

    < 1.33 >  Scalar dtype: [Double (Float64)]
    < 10 >  Scalar dtype: [Int32]
    < (3,4) >  Scalar dtype: [Complex Double (Complex Float64)]



* 2. Consider we want to create a function that can take arbitrary types of arguments (which is very simple in python, but cannot be achieve in c++), with Scalar, we can do:

.. code-block:: c++
    :linenos:

    Scalar generic_func(const std::vector<Scalar> &args){
        // do something here with args

    }


In such way, user can dynamically decide what and which types of variables to pass into the function. 


.. toctree::
    :maxdepth: 1

