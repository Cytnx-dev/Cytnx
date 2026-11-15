Device
--------------

In Cytnx, all the Device properties are handled by **cytnx.Device**.


Number of threads
********************
**Device.Ncpus** is the maximum number of threads that Cytnx's own CPU kernels use. Currently these are the tensor transposes behind ``contiguous()``, which go through HPTT. You can read it:

* In Python:

.. literalinclude:: ../../code/python/doc_codes/guide_Device_Ncpus.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../code/cplusplus/doc_codes/guide_Device_Ncpus.cpp
    :language: c++
    :linenos:

You can also set it, for example ``cytnx.Device.Ncpus = 4`` in Python or ``cytnx::Device.Ncpus = 4`` in C++.

The default is OpenMP's thread count, capped at the number of physical cores. OpenMP's count honours ``OMP_NUM_THREADS`` and the CPUs the process may run on: ``taskset``, cpusets, and batch-system allocations such as Slurm's. If Cytnx is not compiled with OpenMP, Device.Ncpus is 1.

A transpose uses at most one thread per 64 KiB of data, so small transposes always run on a single thread. ``OMP_NUM_THREADS`` and ``omp_set_num_threads()`` also cap the thread count. For example, the following line limits Cytnx's transposes, and any OpenMP-threaded BLAS such as MKL, to 16 threads:

.. code-block:: console

    export OMP_NUM_THREADS=16


.. Note::

    The BLAS/LAPACK library manages its own threads. Set those with the library's own variables, such as ``OPENBLAS_NUM_THREADS`` or ``MKL_NUM_THREADS``. With an OpenBLAS built on pthreads, running OpenBLAS and Cytnx's transposes multithreaded on the same cores at once can be much slower than running either alone.


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
