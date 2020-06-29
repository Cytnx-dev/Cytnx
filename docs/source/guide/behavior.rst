Objects behavior 
------------------

Eveything is reference
************************
To provide a direct translate between C++ API and python, as well as reduce the redundant memory allocation, all the objects (except Accessor and LinOp) in Cytnx especially in C++ side, are **references** just like in python. 

Let's look at the following example. Consider the **Tensor** object in Cytnx

* In python:

.. code-block:: python
    :linenos:

    A = cytnx.Tensor([2,3])
    B = A

    print(B is A) # false


* In c++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::Tensor({2,3});
    auto B = A;

    cout << cytnx::is(B,A) << endl;

Output >>

.. code-block:: text

    True


.. Note::
    
    In python , this is very straight forward. We implement the *is* clause to cytnx C++ API, so that C++ and python can have exactly the same behavior. You can use **is()** to check if two objects are the same. 


clone
*********
In the case where a copy of object is needed, you can use **clone()** to get a copy of objects. 

* In python:

.. code-block:: python 
    :linenos:
    
    A = cytnx.Tensor([2,3]);
    B = A;
    C = A.clone();
    
    print(B is A)
    print(C is A)



* In c++:

.. code-block:: c++
    :linenos:
 
    auto A = cytnx::Tensor({2,3});
    auto B = A;
    auto C = A.clone();
    
    cout << cytnx::is(B,A) << endl;
    cout << cytnx::is(C,A) << endl;

Output>>

.. code-block:: text

    True
    False


.. Note::

    Here, we use **Tensor** as example, but in general all the objects in Cytnx (except Accessor and LinOp) also follows the same behavior, and you can use **is** and **clone** as well. 


.. toctree::

