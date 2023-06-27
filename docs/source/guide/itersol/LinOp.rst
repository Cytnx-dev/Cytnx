LinOp class
--------------
Cytnx provides the ability to define a customized linear operators using the **LinOp** class. 

Before diving into the **LinOp** class, let's take a look at a simple example:

In linear algebra, a linear operator can be represented as a matrix :math:`\boldsymbol{\hat{H}}` that operates on a vector :math:`\boldsymbol{x}`, resulting in an output vector :math:`\boldsymbol{y}` as

.. math::

    y = \hat{H} x


If we consider :math:`\boldsymbol{\hat{H}}` to be a matrix, this matrix-vector multiplication can simply be achieved by using **linalg.Dot**. 

As an example, we multiply a matrix :math:`\boldsymbol{\hat{H}}` with shape (4,4) and a vector :math:`\boldsymbol{x}` with 4 elements: 

* In Python :

.. code-block:: python
    :linenos:

    x = cytnx.ones(4)
    H = cytnx.arange(16).reshape(4,4)

    y = cytnx.linalg.Dot(H,x)

    print(x)
    print(H)
    print(y)


* In C++:

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
    [6.00000e+00 2.20000e+01 3.80000e+01 5.40000e+01 ]





Such an explicit matrix-vector multiplication can be done for small matrices :math:`\boldsymbol{\hat{H}}`. 

**What if the matrix is very large but sparse? Or if more internal structure can be used?**

Clearly, if most of the elements of the matrix are zero, using a dense structure is not only memory-inefficient but also leads to unnecessarily large computational costs as many zeros need to be multiplied and added.  
One way to solve this problem is to use a sparse representation with a pre-defined sparse structure. Indeed, most linear algebra libraries provide these standardized sparse data structures.  

There is, however, a more general way to represent linear operator :math:`\boldsymbol{\hat{H}}`. 
Instead of thinking of it as a matrix and making use of various different internal data structures, we can think of this linear operator :math:`\boldsymbol{\hat{H}}` as a linear **function** that maps an input vector :math:`\boldsymbol{x}` to an output vector :math:`\boldsymbol{y}`.

This is precisely what the LinOp class is designed to do. Users can define the mapping operation from an input vector :math:`\boldsymbol{x}` to an output vector :math:`\boldsymbol{y}` inside the LinOp class.


.. Hint::

    This functionality can also be helpful if the matrix has a known internal structure which can be used to speed up the algorithm. For example, in typical tensor network algorithms, the linear operator is often defined by the contraction of a tensor network. Instead of explicitly doing all the contractions and storing the result in a possibly large matrix :math:`\boldsymbol{\hat{H}}`, it can be much more efficient to contract the tensor network directly with the input tensor :math:`\boldsymbol{x}`. Then, the order of the index summations can be chosen in a way that minimizes the number of operations needed. An example of this is given in the :ref:`SectionDMRG` algorithm. 
    

There are two ways to define a linear operator:

1. Pass a callable function with appropriate signature to the LinOp object.

2. Inherit the LinOp class and overload the **matvec** member function.

Let's consider a simple example of an operator that acts on an input vector :math:`\boldsymbol{x}` with 4 elements. It interchanges the 1st and 4th element, and adds one to both the 2nd and 3rd elements. The output then is again a dim=4 vector :math:`\boldsymbol{y}`. 


Pass a function
*****************
The simplest implementation of a linear operator is to define a function and pass it to the LinOp class. 

First, let's define the function:

* In Python:

.. code-block:: python 
   :linenos:     

    def myfunc(v):
        out = v.clone()
        out[0],out[3] = v[3], v[0] #swap
        out[1]+=1 #add 1
        out[2]+=1 #add 1
        return out
    

* In C++:

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

    The function should have the signature **Tensor f(const Tensor &)** with NO additional arguments. If the linear algebra operator depends on more parameters, the approach in the next section :ref:`Inherit the LinOp class` can be used instead.


Next, we create a **LinOp** object, and pass the function *myfunc* to it. 

* In Python:

.. code-block:: python 
   :linenos:     

    H = cytnx.LinOp("mv",nx=4,\
                    dtype=cytnx.Type.Double,\
                    device=cytnx.Device.cpu,\
                    custom_f=myfunc)
    

* In C++:

.. code-block:: c++
   :linenos:     

    auto H = LinOp("mv",4,Type.Double,
                          Device.cpu,
                          myfunc);

The meaning of the arguments is:

+---+-------------------+--------------------------------------------------------------------+
| 1 | "mv"              | indicates matrix-vector multiplication                             |
+---+-------------------+--------------------------------------------------------------------+
| 2 | nx=4              | sets the input dimension to 4                                      |
+---+-------------------+--------------------------------------------------------------------+
| 3 | dtype=Type.Double | data type of input/output vector of the custom function            |
+---+-------------------+--------------------------------------------------------------------+
| 4 | device=Device.cpu | device type of the input/output vectors of the custom function     |
+---+-------------------+--------------------------------------------------------------------+
| 5 | custom_f=myfunc   | the custom function                                                |
+---+-------------------+--------------------------------------------------------------------+

Finally, we can use this object by calling **LinOp.matvec**:

* In Python:

.. code-block:: python
    :linenos:

    x = cytnx.arange(4)
    y = H.matvec(x)
    print(x)
    print(y)
    
* In C++:

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
Cytnx exposes the interface **LinOp.matvec**, which provides more flexibility for users who want to include additional data/functions associated with the mapping. This can be achieved with inheritance from the **LinOp** class. 

Let's demonstrate this in a similar example as previously. Again, we consider an operator that interchanges the 1st and 4th elements. But this time, we want to add a constant, which is an external parameter, to the 2nd and 3rd elements. 

First, let's create a class that inherits from **LinOp**, with a class member **AddConst**. 

* In Python:

.. code-block:: python
    :linenos:

    class MyOp(cytnx.LinOp):
        AddConst = 1# class member.

        def __init__(self,aconst):
            # here, we fix nx=4, dtype=double on CPU, 
            # so the constructor only takes the external argument 'aconst'
            
            ## Remember to init the mother class. 
            ## Here, we don't specify custom_f!
            LinOp.__init__(self,"mv",4,cytnx.Type.Double,\
                                       cytnx.Device.cpu )

            self.AddConst = aconst

* In C++:

.. code-block:: c++
    :linenos:

    using namespace cytnx;
    class MyOp: public LinOp{
        public:
            double AddConst;

            MyOp(double aconst):
                LinOp("mv",4,Type.Double,Device.cpu){ //invoke base class constructor!

                this->AddConst = aconst;
            }

    };


Next, we need to overload the **matvec** member function, as it defines the mapping from input :math:`\boldsymbol{x}` to the output :math:`\boldsymbol{y}`.

* In Python:

.. code-block:: python
    :linenos:
    :emphasize-lines: 15-20

    class MyOp(cytnx.LinOp):
        AddConst = 1# class member.

        def __init__(self,aconst):
            # here, we fix nx=4, dtype=double on CPU, 
            # so the constructor only takes the external argument 'aconst'
            
            ## Remember to init the mother class. 
            ## Here, we don't specify custom_f!
            cytnx.LinOp.__init__(self,"mv",4,cytnx.Type.Double,\
                                             cytnx.Device.cpu )

            self.AddConst = aconst

        def matvec(self, v):
            out = v.clone()
            out[0],out[3] = v[3],v[0] # swap
            out[1]+=self.AddConst #add constant
            out[2]+=self.AddConst #add constant
            return out


* In C++:

.. code-block:: c++
    :linenos:
    :emphasize-lines: 12-19

    using namespace cytnx;
    class MyOp: public LinOp{
        public:
            double AddConst;

            MyOp(double aconst):
                LinOp("mv",4,Type.Double,Device.cpu){ //invoke base class constructor!

                this->AddConst = aconst;
            }

            Tensor matvec(const Tensor& v) override{
                auto out = v.clone();
                out(0) = v(3); //swap
                out(3) = v(0); //swap
                out[1]+=this->AddConst; //add const
                out[2]+=this->AddConst; //add const
                return out;
            }

    };

Now, the class can be be used. We demonstrate this in in the following and set the constant to be added to 7: 

* In Python:

.. code-block:: python
    :linenos:

    myop = MyOp(7)
    x = cytnx.arange(4)
    y = myop.matvec(x)

    print(x)
    print(y)



* In C++:

.. code-block:: c++
    :linenos:

    auto myop = MyOp(7);
    auto x = cytnx::arange(4);
    auto y = myop.matvec(x);

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
    [3.00000e+00 8.00000e+00 9.00000e+00 0.00000e+00 ]


Ex: sparse data structure with mapping function 
****************************************************
With the flexibility provided by overloading the **matvec** member function, users can actually define their own sparse data structures of an operator. 

As an example, we want to define a sparse matrix :math:`\boldsymbol{A}` with shape=(1000,1000) with ONLY two non-zero elements A[1,100]=4 and A[100,1]=7. All other elements are zero. We do not have to construct a dense tensor with size :math:`10^6`. Instead, we can simply use the **LinOp** class:

* In Python:

.. code-block:: python
    :linenos:

    class Oper(cytnx.LinOp):
        Loc = []
        Val = []

        def __init__(self):
            cytnx.LinOp.__init__(self,"mv",1000)

            self.Loc.append([1,100])
            self.Val.append(4.)

            self.Loc.append([100,1])
            self.Val.append(7.)

        def matvec(self,v):
            out = cytnx.zeros(v.shape(),v.dtype(),v.device())
            for i in range(len(self.Loc)):
                out[self.Loc[i][0]] += v[self.Loc[i][1]]*self.Val[i]
            return out


    A = Oper();
    x = cytnx.arange(1000)
    y = A.matvec(x)

    print(x[1].item(),x[100].item())
    print(y[1].item(),y[100].item())


Output>>

.. code-block:: text
    
    1.0 100.0
    400.0 7.0



.. Hint::

    In this example, we use the python API. C++ can be used similarly. 
    

Prestore/preconstruct sparse elements
****************************************
In the previous example, we showed how to construct a linear operator by overloading the **matvec** member function of the LinOp class. This is straight forward and simple, but in cases where the custom mapping contains many for-loops, handling them in Python is not optimal for performance reasons. 

Since v0.6.3a, the option **"mv_elem"** is available in the constructor of the LinOp class. It allows users to pre-store the indices and values of the non-zero elements, similar to the standard sparse storage structure. If this is used, Cytnx handles the internal structure and optimizes the matvec performance. Again, let's use the previous example: a sparse matrix :math:`\boldsymbol{A}` with shape=(1000,1000) and ONLY two non-zero elements A[1,100]=4 and A[100,1]=7.  

* In Python:

.. code-block:: python 
    :linenos:
    :emphasize-lines: 6,7
    
    class Oper(cytnx.LinOp):

        def __init__(self):
            cytnx.LinOp.__init__(self,"mv_elem",1000)

            self.set_elem(1,100,4.)
            self.set_elem(100,1,7.)

    A = Oper();
    x = cytnx.arange(1000)
    y = A.matvec(x)

    print(x[1].item(),x[100].item())
    print(y[1].item(),y[100].item())


Notice that instead of overloading the **matvec** function, we use the **set_elem** member function in the LinOp class to set the indices and values of the elements. This information is then stored internally in the LinOp class, and we let the LinOp class provide and optimize **matvec**. 

In :ref:`Lanczos solver`, we will see how we can benefit from the LinOp class by passing this object to Cytnx's iterative solver. This way the eigenvalue problem can be solved efficiently with our customized linear operator. 



.. toctree::
