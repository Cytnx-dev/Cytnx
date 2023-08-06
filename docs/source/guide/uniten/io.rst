Save/Load a UniTensor
-----------------
Cyntx provides a way to save/read UniTensors to/from a file.

Save a UniTensor
*****************
To save a Tensor to a file, simply call **UniTensor.Save(filepath)**. 

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_io_Save.py
    :language: python
    :linenos:


This will save UniTensors *T1* and *T2* to the current directory as **Untagged_ut.cytnx** and **sym_ut.cytnx**, with *.cytnx* as file extension.


Load a UniTensor
******************
Now, let's load the UniTensor from the file. 

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_io_Load.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_io_Load.out
    :language: text

In this example we see how the block date and meta information (name, bonds, labels ... ) are kept when we save the UniTensor.

.. toctree::
