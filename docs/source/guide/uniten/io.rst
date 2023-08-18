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
<<<<<<< 9cc4c5ff9162d034cec2483658124c1dc0294a71
=======
    
    # Create an untagged unitensor and save
    T1 = cytnx.UniTensor(cytnx.zeros([4,4]), rowrank=1, labels=["a","b"], name="Untagged_Unitensor")
    T1.Save("Untagged_ut")

    # Create an unitensor with symmetry and save
    bd = cytnx.Bond(cytnx.BD_IN,[[1],[0],[-1]],[1,2,1])
    T2 = cytnx.UniTensor([bd, bd.redirect()], rowrank=1, labels=["a","b"], name="symmetric_Unitensor")
    T2.put_block(cytnx.ones([2,2]),1)
    T2.Save("sym_ut")
>>>>>>> 4962bbf5421223c563daeec3dbad0b3a7c110c3e


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
