Conventions
------------------

These are general conventions that hold for all parts of Cytnx.

Function naming conventions
****************************
Generally, the function naming scheme in Cytnx follows the rules:


1. If the function is **acting on objects** (taking object as arguments), they will start with the first letter being **capical**. Examples are the linalg functions, Contract etc...

    .. code-block:: python

        cytnx.linalg.Svd(A)
        cytnx.linalg.Qr(A)
        cytnx.linalg.Sum(A)

        cytnx.Contract(A,B)


2. If a function is a **member function**, or a **generating function** (such as zeros(), ones() ...), then they usually start with a **lower** case letter, for example:

    .. code-block:: python

        A = cytnx.UniTensor.zeros([2,3,4])

        A.permute(0,2,1)
        B = A.contiguous()


3. **Objects** in Cytnx always start with **capical** letters, for example:

    .. code-block:: python

        A = cytnx.UniTensor()
        B = cytnx.Bond()
        C = cytnx.Network()
        D = cytnx.Tensor()


4. Functions end with **underscore** indicate that the *input* will be changed. For member functions, this is an inplace operation


    .. code-block:: python

        A = cytnx.zeros([2,3,4])

        A.contiguous_() # A gets changed
        B = A.contiguous() # A is not changed, but return a copy B (see Tensor for further info)

        A.permute_(0,2,1) # A gets changed
        C = A.permute(0,2,1) # A is not changed but return a new B as A's permute


Everything is a reference
**************************
To provide a direct translation between the C++ API and Python, as well as to reduce the redundant memory allocation, all the objects (except Accessor and LinOp) in Cytnx are **references**, also in C++, just like in Python.

Let's look at the following example. Consider the **Tensor** object in Cytnx

* In Python:

.. literalinclude:: ../../code/python/doc_codes/guide_behavior_assign.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../code/cplusplus/doc_codes/guide_behavior_assign.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../code/python/outputs/guide_behavior_assign.out
    :language: text

.. Note::

    In Python, this is very straight forward. We implement the *is* clause in the Cytnx C++ API, such that C++ and Python can have exactly the same behavior. You can use **is()** to check if two objects are the same.


clone
*********
In the case where a copy of an object is needed, you can use **clone()**.

* In Python:

.. literalinclude:: ../../code/python/doc_codes/guide_behavior_clone.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../code/cplusplus/doc_codes/guide_behavior_clone.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../code/python/outputs/guide_behavior_clone.out
    :language: text

.. Note::

    Here, we use **Tensor** as an example, but in general all the objects in Cytnx (except Accessor and LinOp) also follow the same behavior, and you can use **is** and **clone** as well.


.. toctree::
