
Device
--------------

In cytnx, all the Device properties are handled by **cytnx.Device**. 


Number of threads
********************
To check how many threads can be used in your current program by cytnx, you can use **Device.Ncpus**. 

* In C++

.. code-block:: c++
    
    cout << Device.Ncpus;


* In python

.. code-block:: python

    print(cytnx.Device.Ncpus)



If cytnx is not compiled with OpenMP avaliable, the Device.Ncpus will always return 1.


.. Note::

    For the parallel with OpenMP, by default mkl use all the available threads. 
    However, any cytnx internal function that utilizes OpenMP may or may not automatically use all threads, depending on your current environment configuration. 


If OpenMP is enable and you want to set restriction on how many threads you want your program to use, this can be done by simply changing the environment variable before you execute your program. For example, the following line will make mkl as well as cytnx internal functions using 16 threads in all places where they are parallellizable. 

.. code-block:: console
    
    export OMP_NUM_THREADS=16



.. Warning::

    Do not change this value manually! 


Number of GPUs
********************
To check how many gpus can be used in your current program by cytnx, you can use **Device.Ngpus**. 

* In C++

.. code-block:: c++

    cout << Device.Ngpus;


* In python

.. code-block:: python

    print(cytnx.Device.Ngpus)


If cytnx is not compiled with CUDA avaliable, the Device.Ngpus will always return 0.


.. Warning::

    Do not change this value manually! 


GPU status
*********************
For system with multi-gpu, cytnx ultilize peer-access feature to transfer data between GPUs when they are avaliable. The **Device.Print_Property()** will list the avaliablility between GPUs.


.. code-block:: python 

    cytnx.Device.Print_Property();


* Output example: 

1. executed on a node with 4 GPUs installed with peer-access avaliable between gpu-id=0 <-> gpu-id=2:

.. code-block:: text
    
    === CUDA support ===
    \: Peer PCIE Access\:
         0  1  2  3
       ------------
     0|  x  0  1  0
     1|  0  x  0  0
     2|  1  0  x  0
     3|  0  0  0  x
    --------------------

    
2. executed when cytnx is not compiled with CUDA:

.. code-block:: text

    === No CUDA support ===







