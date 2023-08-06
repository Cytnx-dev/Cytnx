LinOp class
--------------
Cytnx provides the ability to define a customized linear operators using the **LinOp** class. 

Before diving into the **LinOp** class, let's take a look at a simple example:

In linear algebra, a linear operator can be represented as a matrix :math:`\boldsymbol{\hat{H}}` that operates on a vector :math:`\boldsymbol{x}`, resulting in an output vector :math:`\boldsymbol{y}` as

.. math::

    y = \hat{H} x


If we consider :math:`\boldsymbol{\hat{H}}` to be a matrix, this matrix-vector multiplication can simply be achieved by using **linalg.Dot**. 

As an example, we multiply a matrix :math:`\boldsymbol{\hat{H}}` with shape (4,4) and a vector :math:`\boldsymbol{x}` with 4 elements: 

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_itersol_LinOp_Dot.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_itersol_LinOp_Dot.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_itersol_LinOp_Dot.out
    :language: text


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


Inherit the LinOp class
************************
Cytnx exposes the interface **LinOp.matvec**, which provides more flexibility for users who want to include additional data/functions associated with the mapping. This can be achieved with inheritance from the **LinOp** class. 

Let's demonstrate this in a similar example as previously. Again, we consider an operator that interchanges the 1st and 4th elements. But this time, we want to add a constant, which is an external parameter, to the 2nd and 3rd elements. 

First, let's create a class that inherits from **LinOp**, with a class member **AddConst**. 

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_itersol_LinOp_inherit.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_itersol_LinOp_inherit.cpp
    :language: c++
    :linenos:


Next, we need to overload the **matvec** member function, as it defines the mapping from input :math:`\boldsymbol{x}` to the output :math:`\boldsymbol{y}`.

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_itersol_LinOp_matvec.py
    :language: python
    :linenos:
    :emphasize-lines: 15-20

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_itersol_LinOp_matvec.cpp
    :language: c++
    :linenos:
    :emphasize-lines: 10-17


Now, the class can be be used. We demonstrate this in in the following and set the constant to be added to 7: 

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_itersol_LinOp_demo.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_itersol_LinOp_demo.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_itersol_LinOp_demo.out
    :language: text


Example: sparse data structure with mapping function 
*****************************************************
With the flexibility provided by overloading the **matvec** member function, users can actually define their own sparse data structures of an operator. 

As an example, we want to define a sparse matrix :math:`\boldsymbol{A}` with shape=(1000,1000) with ONLY two non-zero elements A[1,100]=4 and A[100,1]=7. All other elements are zero. We do not have to construct a dense tensor with size :math:`10^6`. Instead, we can simply use the **LinOp** class:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_itersol_LinOp_sparse_mv.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_itersol_LinOp_sparse_mv.out
    :language: text


.. Hint::

    In this example, we use the python API. C++ can be used similarly. 
    

Prestore/preconstruct sparse elements
****************************************
In the previous example, we showed how to construct a linear operator by overloading the **matvec** member function of the LinOp class. This is straight forward and simple, but in cases where the custom mapping contains many for-loops, handling them in Python is not optimal for performance reasons. 

Since v0.6.3a, the option **"mv_elem"** is available in the constructor of the LinOp class. It allows users to pre-store the indices and values of the non-zero elements, similar to the standard sparse storage structure. If this is used, Cytnx handles the internal structure and optimizes the matvec performance. Again, let's use the previous example: a sparse matrix :math:`\boldsymbol{A}` with shape=(1000,1000) and ONLY two non-zero elements A[1,100]=4 and A[100,1]=7.  

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_itersol_LinOp_sparse_mv_elem.py
    :language: python
    :linenos:
    :emphasize-lines: 6-7

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_itersol_LinOp_sparse_mv_elem.out
    :language: text

Notice that instead of overloading the **matvec** function, we use the **set_elem** member function in the LinOp class to set the indices and values of the elements. This information is then stored internally in the LinOp class, and we let the LinOp class provide and optimize **matvec**. 

In :ref:`Lanczos solver`, we will see how we can benefit from the LinOp class by passing this object to Cytnx's iterative solver. This way the eigenvalue problem can be solved efficiently with our customized linear operator. 



.. toctree::
