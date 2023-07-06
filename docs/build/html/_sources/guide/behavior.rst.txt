Objects behavior 
------------------

Everything is reference
************************
To provide a direct translation between the C++ API and Python, as well as to reduce the redundant memory allocation, all the objects (except Accessor and LinOp) in Cytnx are **references**, also in C++, just like in Python. 

Let's look at the following example. Consider the **Tensor** object in Cytnx

* In Python:

.. code-block:: python
    :linenos:

    A = cytnx.Tensor([2,3])
    B = A

    print(B is A) # true


* In C++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::Tensor({2,3});
    auto B = A;

    cout << cytnx::is(B,A) << endl;

Output >>

.. code-block:: text

    True


.. Note::
    
    In Python, this is very straight forward. We implement the *is* clause in the Cytnx C++ API, such that C++ and Python can have exactly the same behavior. You can use **is()** to check if two objects are the same. 


clone
*********
In the case where a copy of an object is needed, you can use **clone()**. 

* In Python:

.. code-block:: python 
    :linenos:
    
    A = cytnx.Tensor([2,3]);
    B = A;
    C = A.clone();
    
    print(B is A)
    print(C is A)



* In C++:

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

    Here, we use **Tensor** as an example, but in general all the objects in Cytnx (except Accessor and LinOp) also follow the same behavior, and you can use **is** and **clone** as well. 


.. toctree::

