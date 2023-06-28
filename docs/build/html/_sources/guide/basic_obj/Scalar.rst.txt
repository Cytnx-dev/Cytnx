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
    
    1. We can also just assign a C++ type to Scalar as in the case of **A2**. 

    2. In the case of variable **A3**, the assigned value *10* by default in C++ is an integer type. If we want the Scalar to be of type *double*, we can pass an additional dtype to further specify the type explicitly. 

* 3. Convert to a C++ data type:

    Converting to a C++ data type works just like other type castings in C++:


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

    Note the slightly different syntax in case of complex type castings. Use **complex128()** and **complex64()** to convert to a standard C++ type **complex<double>** and **complex<float>** respectively.


Change data type
******************
    To change the data type of a Scalar, use **.astype()**. 


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
The Scalar type allows for many possibilities in C++ that are otherwise hard to achieve. We will consider some examples in the following.

* 1. If we want to have a list (vector) with elements having different data types, we can use *tuple* objects in C++. However, that requires the number of elements and data types to be known and fixed a priori. Using Scalar, we can also create a vector with a variable number of elements and variable data types:

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



* 2. In C++, unlike Python, we can not create a function that can take arbitrary types of arguments. This becomes possible with Scalar:

.. code-block:: c++
    :linenos:

    Scalar generic_func(const std::vector<Scalar> &args){
        // do something here with args

    }


This way, the user can dynamically decide which types of variables to pass to the function. 


.. toctree::
    :maxdepth: 1

