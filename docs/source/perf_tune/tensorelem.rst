Access single element of Tensor in C++
****************************************
In Tensor introduction section, we mentioned that there are in general two ways to access element in a Tensor: **Tensor(i,j,k,...)** or **Tensor.at({i,j,k....})**. 
Here, we give a general guide on their difference and when you should use one instead of the other. 



Let's consider the straightforward way to do access via operator().

The benefit of using this is that it is general, meaning that you can put Accessor, integer or String for the argument. 

For example:

.. code-block:: c++
    :linenos:

    A = cytnx::arange(30).reshape(3,5,2);
    A(":",Accessor::range(0,4,1),0);
    

In the scnario where we want to access only single element, using operator(), one can do:

.. code-block:: c++
    :linenos:

    Tensor out = A(0,1,0);

Note that here, the return is a **Tensor**, not a **Scalar**. Cytnx view this access as slicing Tensor wiwth dim(1,1,1). 

.. hint::
    
    For Tensor with single element, one can use A.item() to get the element as Scalar, or use A.item<>() to get standard C++ type via template.  
    
Although this is legit, it will have more overhead than we want. Since lots of meta data is being calculated in background. If we want to frequently access elements, it is better to use **at()**:

.. code-block:: c++
    :linenos:

    Scalar out = A.at({0,1,0});
    

The return of **at()** is directly a Scalar. Further more, it has less overhead than **operator()** in the case of single element accessing. 


* Result for comparing two method on a Tensor with shape(3,4,5), with repeatly 100k times single element accessing: 

.. code-block:: text
    
    R = 3.2337x faster





