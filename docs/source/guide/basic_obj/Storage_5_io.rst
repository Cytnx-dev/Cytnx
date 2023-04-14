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

.. code-block:: c++
    :linenos:

    auto A = cytnx::Storage(4);
    A.fill(6);
    A.Save("S1");

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


save & load from/to binary
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

.. code-block:: c++
    :linenos:

    // read
    auto A = cytnx::Storage(10);
    A.fill(10);
    cout << A << endl;

    A.Tofile("S1");

    //load
    auto B = cytnx::Storage::Fromfile("S1",cytnx::Type.Double);
        
    cout << B << endl;

Output>>

.. code-block:: text

    dtype : Double (Float64)
    device: cytnx device: CPU
    size  : 10
    [ 1.00000e+01 1.00000e+01 1.00000e+01 1.00000e+01 1.00000e+01 1.00000e+01 1.00000e+01 1.00000e+01 1.00000e+01 1.00000e+01 ]

    dtype : Double (Float64)
    device: cytnx device: CPU
    size  : 10
    [ 1.00000e+01 1.00000e+01 1.00000e+01 1.00000e+01 1.00000e+01 1.00000e+01 1.00000e+01 1.00000e+01 1.00000e+01 1.00000e+01 ]


.. Note:: 

    You can also choose to load a part of the file with an additional argument *count* when using **Fromfile**





.. toctree::
