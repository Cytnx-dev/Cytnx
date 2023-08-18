Save/Load a Tensor
-----------------
Cyntx provides a way to save/read Tensors to/from a file.

Save a Tensor
*****************
To save a Tensor to a file, simply call **Tensor.Save(filepath)**. 

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Tensor_7_io_Save.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Tensor_7_io_Save.cpp
    :language: c++
    :linenos:

This will save Tensor *A* to the current directory as **T1.cytn**, with *.cytn* as file extension.


Load a Tensor
******************
Now, let's load the Tensor from the file. 

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Tensor_7_io_Load.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Tensor_7_io_Load.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Tensor_7_io_Load.out
    :language: text


.. toctree::
