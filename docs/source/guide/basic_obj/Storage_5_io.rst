Save/Load a storage
--------------------
We can save/read a Storage instance to/from a file.

Save a Storage
*****************
To save a Storage to file, simply call **Storage.Save(filepath)**.

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Storage_5_io_Save.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Storage_5_io_Save.cpp
    :language: c++
    :linenos:

This will save Storage *A* to the current directory as **S1.cyst**.

.. Tip::

    The common file extension for a Storage is *.cyst*.

.. warning::

    The file extension should be explicitly added. The previous behavior of attaching the extension *.cyst* automatically will be deprecated.

Load a Storage
******************
Now, let's load the Storage from the file.

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Storage_5_io_Load.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Storage_5_io_Load.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Storage_5_io_Load.out
    :language: text


Save & load from/to binary
**************************
We can also save all the elements in a binary file without any additional header information associated to the storage format. This is similar to numpy.tofile/numpy.fromfile.

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Storage_5_io_from_to_file.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Storage_5_io_from_to_file.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Storage_5_io_from_to_file.out
    :language: text

.. Note::

    You can also choose to load a part of the file with an additional argument *count* when using **Fromfile**


.. toctree::
