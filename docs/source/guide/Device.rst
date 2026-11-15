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

Cytnx's CPU kernels are single-threaded unless you ask for more, so the default is 1. To use more threads, set **Device.Ncpus**, for example ``cytnx.Device.Ncpus = 4`` in Python or ``cytnx::Device.Ncpus = 4`` in C++, or set the environment variable before starting the program:

.. code-block:: console

    export CYTNX_NUM_THREADS=4

With more than one thread allowed, a transpose uses at most one thread per 64 KiB of data, so small transposes still run on a single thread. ``OMP_NUM_THREADS``, ``omp_set_num_threads()`` and the CPUs the process may run on (``taskset``, cpusets, and batch-system allocations such as Slurm's) cap the thread count as well. If Cytnx is not compiled with OpenMP, transposes are always single-threaded.

Multi-threaded transposes pay off mainly for large transposes run in isolation. In DMRG-style workloads, where transposes alternate with BLAS calls, single-threaded transposes measured faster; see `#939 <https://github.com/Cytnx-dev/Cytnx/issues/939>`_.


.. Note::

    The BLAS/LAPACK library manages its own threads. Set those with the library's own variables, such as ``OPENBLAS_NUM_THREADS`` or ``MKL_NUM_THREADS``.

    If you raise **Device.Ncpus**, also make idle threads sleep instead of spinning: set ``OMP_WAIT_POLICY=passive`` and, with an OpenBLAS built on pthreads, ``OPENBLAS_THREAD_TIMEOUT=4``. Otherwise idle OpenMP and OpenBLAS threads spin and compete for the same cores. For a one-site DMRG effective Hamiltonian at bond dimension 512 on a 32-core Zen 5, ``CYTNX_NUM_THREADS=32`` made each application about 1.6 times faster than the default with these settings, and about 5 times slower without them.


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
