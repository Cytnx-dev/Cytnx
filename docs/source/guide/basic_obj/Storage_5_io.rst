Save/Load
-----------------
We can save/read Storage to/from file.

Save a Storage
*****************
To save a Storage to file, simply call **Storage.Save(filepath)**. 

* In python:

.. code-block:: python 
    :linenos:
    
    A = cytnx.Storage(4)
    A.fill(6)
    A.Save("S1")


* In c++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::Storage(4);
    A.fill(6);
    A.Save("S1");

This will save Storage *A* to the current directory as **T1.cyst**, with extension *.cyst*


Load a Storage
******************
Now, let's try to load the Storage from the file. 

* In python:

.. code-block:: python
    :linenos:
    
    A = cytnx.Storage.Load("S1.cyst")
    print(A)

* In c++:

.. code-block:: c++
    :linenos:
    
    auto A = cytnx::Storage::Load("S1.cyst");
    cout << A << endl;

Output>>

.. code-block:: text

    dtype : Double (Float64)
    device: cytnx device: CPU
    size  : 4
    [ 6.00000e+00 6.00000e+00 6.00000e+00 6.00000e+00 ]


.. toctree::
