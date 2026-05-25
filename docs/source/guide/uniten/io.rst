Save/Load a UniTensor
-------------------------

A UniTensor can be saved to and loaded from a file.

Save a UniTensor
*****************
To save a UniTensor to a file, simply call **UniTensor.Save(filepath)**.

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_io_Save.py
    :language: python
    :linenos:

This saves UniTensors *T1* and *T2* in the current directory as **Untagged_ut.cytnx** and **sym_ut.cytnx**.

.. Tip::

    The common file extension for a UniTensor is *.cytnx*.

.. warning::

    The file extension should be explicitly added. The previous behavior of attaching the extension *.cytnx* automatically will be deprecated.


Load a UniTensor
******************
Similarly, a UniTensor can be loaded from a file:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_io_Load.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_io_Load.out
    :language: text

In this example we see how the block data and meta information (name, bonds, labels ... ) are restored when saving and loading UniTensor.

.. toctree::
