Device
--------------

In Cytnx, all the Device properties are handled by **cytnx.Device**.


CPU thread settings
********************

Cytnx delegates floating-point CPU transposes to HPTT. HPTT follows the current
OpenMP thread settings when it executes. Set ``OMP_NUM_THREADS`` before starting
your program to select the default for HPTT and other OpenMP libraries:

.. code-block:: console

    OMP_NUM_THREADS=1 python simulation.py

A larger value permits parallel transposes. Nested lists such as
``OMP_NUM_THREADS=4,1`` and later C++ ``omp_set_num_threads()`` calls are respected.
OpenMP's nesting, dynamic-team and thread-limit settings still apply. With no
explicit setting, the OpenMP runtime chooses its default.

Use ``HPTT_NUM_THREADS`` to request a different count for transposes while
retaining the general OpenMP setting for other operations. For example:

.. code-block:: console

    OMP_NUM_THREADS=8 HPTT_NUM_THREADS=1 python simulation.py

``HPTT_NUM_THREADS`` must be a positive integer. HPTT reads it whenever Cytnx
creates a transpose plan, so Python can also set ``os.environ["HPTT_NUM_THREADS"]``
before a subsequent transpose. Invalid values raise ``cytnx.CytnxError`` when
that HPTT operation is requested. Integer tensors and builds without HPTT use
Cytnx's serial transpose implementation.

BLAS libraries retain their own thread controls. OpenMP-threaded BLAS can share
the ``OMP_NUM_THREADS`` default; overrides such as ``MKL_NUM_THREADS`` and
``OPENBLAS_NUM_THREADS`` apply only to their respective libraries. For an
OpenBLAS pthreads build, set ``OPENBLAS_NUM_THREADS`` separately when needed.
See :doc:`../Perf_tune` for CPU thread and wait-policy configuration.

.. note::

    ``Device.Ncpus`` has been removed from the C++ and Python APIs. Replace reads
    or assignments with the threading settings above. The old Python attribute
    was a copy of the C++ value, so assigning it did not control library threads.
    C++ consumers must rebuild after this public class-layout change.


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
