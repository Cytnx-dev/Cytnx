LinOp class
--------------
Cytnx provide a way for user to customize a linear operator with **LinOp** class. 

Before dive into the **LinOp** class, let's first take a look at a simple example:

In linear algebra, a linear operator can be think as a matrix :math:`\boldsymbol{\hat{H}}` that operate on a vector :math:`\boldsymbol{x}`, resulting as a output vector :math:`\boldsymbol{y}` as

.. math::

    y = \hat{H} x


If we consider :math:`\boldsymbol{\hat{H}}` to a matrix, this matrix-vector multiplication can be simply achieve using **linalg.Dot**. 
Let's consider a vector :math:`\boldsymbol{x}` with dimension 4, and matrix :math:`\boldsymbol{\hat{H}}` with shape (4,4) as 

* In python :

.. code-block:: python
    :linenos:

    x = cytnx.ones(4)
    H = cytnx.arange(16).reshape(4,4)

    y = cytnx.linalg.Dot(H,x)

    print(x)
    print(H)
    print(y)


* In c++:

.. code-block:: c++
    :linenos:
    
    auto x = cytnx::ones(4);
    auto H = cytnx.arange(16).reshape(4,4);

    auto y = cytnx.linalg.Dot(H,x);

    cout << x << endl;
    cout << H << endl;
    cout << y << endl;

Output>>

.. code-block:: text

    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4)
    [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]


    Total elem: 16
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4,4)
    [[0.00000e+00 1.00000e+00 2.00000e+00 3.00000e+00 ]
     [4.00000e+00 5.00000e+00 6.00000e+00 7.00000e+00 ]
     [8.00000e+00 9.00000e+00 1.00000e+01 1.10000e+01 ]
     [1.20000e+01 1.30000e+01 1.40000e+01 1.50000e+01 ]]


    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4)
    [2.40000e+01 2.80000e+01 3.20000e+01 3.60000e+01 ]


The above example consider a matrix :math:`\boldsymbol{\hat{H}}` that is small, with shape (4,4). 

**What if the matrix is very large but sparse?**

Clearly using dense structure is not only very memory insufficient, but also post large computational cost as most of the elements are zero. 
One way to solve such problem is to use sparse representation with pre-defined sparse structure. Indeed, most of the linear algebra library does provide those standardized sparse data structure. 

There is, however, a more general way to represent linear operator :math:`\boldsymbol{\hat{H}}`. 
Instead of think it as a matrix, and defines different internal data structure, we can think this linear operator :math:`\boldsymbol{\hat{H}}` as a **function** that maps the input vector :math:`\boldsymbol{x}` to a output vector :math:`\boldsymbol{y}`. 

This is exactly what LinOp is designed to do. User can define the mapping operation from input :math:`\boldsymbol{x}` to :math:`\boldsymbol{y}` inside LinOp. 

There are two ways we can define a linear operator: 1. Pass a callable function with proper signature to LinOp object. 2. Inherit the LinOp class, and overload the **matvec** member function.

Let's consider a simple example of a operator that operate on a input vector :math:`\boldsymbol{x}` with dimension 4 that interchange the 1st and 4th elements and add one to both the 2nd and 3rd elements. Then output a dim=4 vector math:`\boldsymbol{y}`. 


Pass a function
*****************
The simplest way is to define a function and pass it into LinOp class. 

First, let's define the function:

* In python:

.. code-block:: python 
   :linenos:     

    def myfunc(v):
        out = v.clone()
        out[0],out[3] = v[3], v[0] #swap
        out[1]+=1 #add 1
        out[2]+=1 #add 1
        return out
    

* In c++:

.. code-block:: c++
   :linenos:     

    using namespace cytnx;
    Tensor myfunc(const Tensor &v){
        Tensor out = v.clone();
        out(0) = v(3); //swap
        out(3) = v(0); //swap
        out[1]+=1; //add 1
        out[2]+=1; //add 1
        return out;
    }

.. Note::

    The function should have signature **Tensor f(const Tensor &)** with NO additional argument. Thus it is less flexible if additional arguments are required. In such case, See next section *Inherit the LinOp class* instead.


Next, we create a **LinOp** object, and pass this *myfunc* into it. 

.. code-block:: python 
   :linenos:     

    H = cytnx.LinOp("mv",nx=4,\
                    dtype=cytnx.Type.Double,\
                    device=cytnx.Device.cpu,\
                    custom_f=myfunc)
    

* In c++:

.. code-block:: c++
   :linenos:     

    auto H = LinOp("mv",4,Type.Double,
                          Device.cpu,
                          myfunc);

The meaning of the arguments are:

+---+-------------------+--------------------------------------------------------------------+
| 1 | "mv"              | indicate matrix-vector multiplication                              |
+---+-------------------+--------------------------------------------------------------------+
| 2 | nx=4              | indicate the input dimension = 4                                   |
+---+-------------------+--------------------------------------------------------------------+
| 3 | dtype=Type.Double | indicate the data type of input/output vector of custom function   |
+---+-------------------+--------------------------------------------------------------------+
| 4 | device=Device.cpu | indicate the device type of input/output vector of custom function |
+---+-------------------+--------------------------------------------------------------------+
| 5 | custom_f=myfunc   | the custom funtion.                                                |
+---+-------------------+--------------------------------------------------------------------+

Finally, we can use this object by calling **LinOp.matvec**:

* In python:

.. code-block:: python
    :linenos:

    x = cytnx.arange(4)
    y = H.matvec(x)
    print(x)
    print(y)
    
* In c++:

.. code-block:: c++
    :linenos:

    auto x = cytnx::arange(4);
    auto y = H.matvec(x);
    cout << x << endl;
    cout << y << endl;
    
    
Output>>

.. code-block:: text

    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4)
    [0.00000e+00 1.00000e+00 2.00000e+00 3.00000e+00 ]


    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4)
    [3.00000e+00 2.00000e+00 3.00000e+00 0.00000e+00 ]




Inherit the LinOp class
************************






.. toctree::
