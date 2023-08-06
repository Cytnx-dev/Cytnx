Save/Load a Tensor
-----------------
Cyntx provides a way to save/read Tensors to/from a file.

Save a Tensor
*****************
To save a Tensor to a file, simply call **Tensor.Save(filepath)**. 

* In Python:

.. code-block:: python 
    :linenos:
    
    A = cytnx.arange(12).reshape(3,4)
    A.Save("T1")


* In C++:

.. literalinclude:: ../../../code/cplusplus/guide_codes/3_7_1_ex1.cpp
    :language: c++
    :linenos:

This will save Tensor *A* to the current directory as **T1.cytn**, with *.cytn* as file extension.


Load a Tensor
******************
Now, let's load the Tensor from the file. 

* In Python:

.. code-block:: python
    :linenos:
    
    A = cytnx.Tensor.Load("T1.cytn")
    print(A)

* In C++:

.. literalinclude:: ../../../code/cplusplus/guide_codes/3_7_2_ex1.cpp
    :language: c++
    :linenos:

Output>>

.. literalinclude:: ../../../code/cplusplus/outputs/3_7_2_ex1.out
    :language: text

.. toctree::
