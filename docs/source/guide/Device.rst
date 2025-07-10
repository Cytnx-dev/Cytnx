Device
--------------

In Cytnx, all the Device properties are handled by **cytnx.Device**.


Number of threads
********************
To check how many threads can be used in your current program by Cytnx, you can use **Device.Ncpus**.

* In Python:

.. literalinclude:: ../../code/python/doc_codes/guide_Device_Ncpus.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../code/cplusplus/doc_codes/guide_Device_Ncpus.cpp
    :language: c++
    :linenos:

If Cytnx is not compiled with OpenMP avaliable, the Device.Ncpus will always return 1.


.. Note::

    For parallel computing with OpenMP, by default mkl uses all the available threads.
    However, any Cytnx internal function that utilizes OpenMP may or may not automatically use all threads, depending on your current environment configuration.


If OpenMP is enabled and you want to set a restriction on how many threads you want your program to use, this can be done by simply changing an environment variable before you execute your program. For example, the following line will make mkl as well as Cytnx internal functions use 16 threads in all places where they are parallelizable.

.. code-block:: console

    export OMP_NUM_THREADS=16



.. Warning::

    Do not change this value manually!


Number of GPUs
********************
To check how many GPUs can be used in your current program by Cytnx, you can use **Device.Ngpus**.

* In Python:

.. literalinclude:: ../../code/python/doc_codes/guide_Device_Ngpus.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../code/cplusplus/doc_codes/guide_Device_Ngpus.cpp
    :language: c++
    :linenos:

If Cytnx is not compiled with CUDA available, the Device.Ngpus will always return 0.


.. Warning::

    Do not change this value manually!


GPU status
*********************
For systems with multi-gpu, Cytnx utilizes the peer-access feature to transfer data between GPUs when they are available. The **Device.Print_Property()** will list the availability of GPUs.

.. literalinclude:: ../../code/python/doc_codes/guide_Device_property.py
    :language: python

Output example>>

1. Executed on a node with 4 GPUs installed with peer-access available between gpu-id=0 <-> gpu-id=2:

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


2. Executed when Cytnx is not compiled with CUDA:

.. code-block:: text

    === No CUDA support ===

Initializing tensors on GPU
******************************
How a tensor is created on the GPU is explained in :ref:`Tensor with different dtype and device`. A tensor can also be moved between devices, see :ref:`Transfer between devices`.
