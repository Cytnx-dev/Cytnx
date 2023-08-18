Objects behavior 
------------------

Everything is reference
************************
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

