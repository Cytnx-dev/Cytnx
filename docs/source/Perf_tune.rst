Performance tuning
=====================


.. toctree::
    :maxdepth: 3

    perf_tune/tensordot.rst
    perf_tune/tensorelem.rst


CPU transpose and BLAS threads
------------------------------

HPTT follows the current OpenMP thread request. Use ``OMP_NUM_THREADS=1`` for
serial HPTT transposes, or a larger value to select a shared OpenMP default.
``HPTT_NUM_THREADS=1`` keeps transposes serial independently of that default.
BLAS-specific overrides such as ``MKL_NUM_THREADS`` or ``OPENBLAS_NUM_THREADS``
control their own library and do not set HPTT's count.

For applications that alternate HPTT and threaded BLAS, ``OMP_WAIT_POLICY=passive``
requests that idle OpenMP workers wait passively. An OpenBLAS pthreads build has
its own pool and its own ``OPENBLAS_THREAD_TIMEOUT`` setting. Configure these
separately from the thread counts, and measure the application workload when
choosing settings. Running several processes (including parallel tests) also
requires dividing the available CPUs between those processes.
