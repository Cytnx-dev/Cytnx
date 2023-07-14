Save/Load
-----------------
We can save/read a Storage instance to/from a file.

Save a Storage
*****************
To save a Storage to file, simply call **Storage.Save(filepath)**. 

* In Python:

.. code-block:: python 
    :linenos:
    
    A = cytnx.Storage(4)
    A.fill(6)
    A.Save("S1")


* In C++:

.. literalinclude:: ../../../code/cplusplus/guide_codes/4_5_1_ex1.cpp
    :language: c++
    :linenos:

This will save Storage *A* to the current directory as **T1.cyst**, with extension *.cyst*


Load a Storage
******************
Now, let's load the Storage from the file. 

* In Python:

.. code-block:: python
    :linenos:
    
    A = cytnx.Storage.Load("S1.cyst")
    print(A)

* In C++:

.. literalinclude:: ../../../code/cplusplus/guide_codes/4_5_2_ex1.cpp
    :language: c++
    :linenos:

Output>>

.. literalinclude:: ../../../code/cplusplus/outputs/4_5_2_ex1.out
    :language: text

Save & load from/to binary
**************************
We can also save all the elements in a binary file without any additional header information associated to the storage format. This is similar to numpy.tofile/numpy.fromfile.

* In Python:

.. code-block:: python
    :linenos:

    # read
    A = cytnx.Storage(10);
    A.fill(10);
    print(A);

    A.Tofile("S1");

    #load
    B = cytnx.Storage.Fromfile("S1",cytnx.Type.Double);
    print(B);

* In C++

.. literalinclude:: ../../../code/cplusplus/guide_codes/4_5_3_ex1.cpp
    :language: c++
    :linenos:

Output>>

.. literalinclude:: ../../../code/cplusplus/outputs/4_5_3_ex1.out
    :language: text

.. Note:: 

    You can also choose to load a part of the file with an additional argument *count* when using **Fromfile**





.. toctree::
