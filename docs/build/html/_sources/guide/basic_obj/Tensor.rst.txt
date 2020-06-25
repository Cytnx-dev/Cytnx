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


Other options such as **arange()** (similar as np.arange), and **ones** (similar as np.ones) can also be done. 

* In pyhton : 

.. code-block:: python 
    :linenos:

    cytnx.arange(10);     #rank-1 Tensor from [0,10) with step 1
    cytnx.arange(0,10,2); #rank-1 Tensor from [0,10) with step 2
    cytnx.ones({3,4,5});  #Tensor of shape (3,4,5) with all elements set to one.

* In c++:

.. code-block:: c++
    :linenos:

    cytnx::arange(10);     //rank-1 Tensor from [0,10) with step 1
    cytnx::arange(0,10,2); //rank-1 Tensor from [0,10) with step 2
    cytnx::ones([3,4,5]);  //Tensor of shape (3,4,5) with all elements set to one.


.. Note::

    In cytnx, the conversion of python list will equivalent to C++ **vector** or in some case like here, it is a *initializer list*. 

    The conversion is pretty straight forward, one simple replace [] in python with {}, and you are all set!



.. toctree::

