Tensor
==========
Tensor is the basic building block of Cytnx, just like numpy.array or torch.tensor. 
In fact, the API of Tensor is almost the same as torch.tensor. 

Let's take a look on how to use it.


Define a Tensor
----------------
Just like numpy.array / torch.tensor, Tensor is generally created using generator such as **zero()**, **arange()**, **ones()**.

For example, suppose we want to define a rank-3 tensor with shape (3,4,5), and initialize all elements with zero:

* In python:

.. code-block:: python
    :linenos:

    A = cytnx.zeros([3,4,5]);
        

* In c++:

.. code-block:: c++
    :linenos:

    cytnx::Tensor A = cytnx::zeros({3,4,5});

.. Note::

    1. In cytnx, the conversion of python list will equivalent to C++ *vector* or in some case like here, it is a *initializer list*. 

    2. The conversion in between is pretty straight forward, one simply replace [] in python with {}, and you are all set!



Other options such as **arange()** (similar as np.arange), and **ones** (similar as np.ones) can also be done. 

* In python : 

.. code-block:: python 
    :linenos:

    A = cytnx.arange(10);     #rank-1 Tensor from [0,10) with step 1
    B = cytnx.arange(0,10,2); #rank-1 Tensor from [0,10) with step 2
    C = cytnx.ones({3,4,5});  #Tensor of shape (3,4,5) with all elements set to one.

* In c++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::arange(10);     //rank-1 Tensor from [0,10) with step 1
    auto B = cytnx::arange(0,10,2); //rank-1 Tensor from [0,10) with step 2
    auto C = cytnx::ones([3,4,5]);  //Tensor of shape (3,4,5) with all elements set to one.

:Tips: In C++, you could make use of *auto* to simplify your code! 




.. toctree::

