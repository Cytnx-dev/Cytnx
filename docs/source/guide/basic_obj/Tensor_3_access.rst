Accessing elements
----------------------
Next, let's take a look on how we can access elements of a Tensor.

Get elements 
***************************
On the Python side, we can simply use *slice* to get the elements, just as common with list/numpy.array/torch.tensor in Python. See :numpy-slice:`This page <>` for more details.
In C++, Cytnx ports this approach from Python to the C++ API. You can simply use a **slice string** to access elements. 

For example:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Tensor_3_access_slice_get.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Tensor_3_access_slice_get.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Tensor_3_access_slice_get.out
    :language: text

.. Note::

    1. To convert between Python and C++ APIs, notice that in C++ you need to use operator() instead of operator[] if you are using slice strings to access elements. 
    2. The return value will always be a Tensor object, even it only contains one element.


In the case where you have only one element in a Tensor, you can use **item()** to get the element as a standard Python/C++ type. 

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Tensor_3_access_item.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Tensor_3_access_item.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Tensor_3_access_item.out
    :language: text

.. Note::
    
    1. In C++, using **item<>()** to get the element requires to explicitly specify the type that matches the dtype of the Tensor. If the type specifier does not match, an error will be prompted. 
    2. Starting from v0.7+, users can use item() in C++ without explicitly specifying the type with a template. 


Set elements
***************************
Setting elements is pretty much the same as in numpy.array/torch.tensor. You can assign a Tensor to a specific slice, or set all the elements in that slice to be the same value. 

For example:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Tensor_3_access_slice_set.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Tensor_3_access_slice_set.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Tensor_3_access_slice_set.out
    :language: text

Low-level API (C++ only) 
*******************************
On the C++ side, Cytnx provides lower-level APIs with slightly smaller overhead for getting elements. 
These low-level APIs require using an **Accessor** object. 

* Accessor:
    **Accessor** object is equivalent to Python *slice*. It is sometimes convenient to use aliases to simplify the expression when using it.
    
.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Tensor_3_access_c_accessor.cpp
    :language: c++
    :linenos:


In the following, let's see how it can be used to get/set the elements from/in a Tensor.

1. operator[] (middle level API) :
    
.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Tensor_3_access_c_operator.cpp
    :language: c++
    :linenos:

.. Note::

    Remember to write a braket{} around the elements to be accessed. This is needed because the C++ operator[] can only accept one argument. 


2. get/set (low level API) :
    get() and set() are part of the low-level API. Operator() and Operator[] are all built based on these.
    
.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Tensor_3_access_c_get_set.cpp
    :language: c++
    :linenos:

.. Hint::

    1. Similarly, you can also pass a C++ *vector<cytnx_int64>* as an argument. 

.. Tip::

    If your code makes frequent use of get/set elements, using the low-level API can reduce the overhead.

.. toctree::
