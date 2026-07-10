Install & Usage of Cytnx
============================
To install Cytnx, we recommend user to use anaconda/miniconda to install. However, advanced users can also build from source if necessary.
Both the conda package and the PyPI wheel install only the Python API; the C++ headers and library are not included. See `Using the C++ API`_ below for how to build against Cytnx's C++ library instead.


Conda install
********************
In the following we show how to install Cytnx with conda.

1. Install :anaconda:`Anaconda3 <>` / :miniconda:`Miniconda3 <>`.


2. Create a virtual environment:

* For linux/WSL:

.. code-block:: shell
    :linenos:

    $conda config --add channels conda-forge
    $conda create --channel conda-forge --name cytnx python=3.9 _openmp_mutex=*=*_llvm

.. Note::

    1. We do not support a native Windows package at this stage. If you are using Windows OS, please use WSL.
    2. Currently, the supported Python versions are updated to: linux --  3.9+; MacOS-osx64 -- 3.9+ (no conda support). You can change the python=* argument to the version you like.


* For MacOS:

    Please build from source, currently 0.9+ does not have conda package support.

..
    * For MacOS (non-arm arch)
    .. code-block:: shell
        :linenos:

        $conda config --add channels conda-forge
        $conda create --channel conda-forge --name cytnx python=3.9 llvm-openmp


.. note::

    * See :virtualenv:`This page <>` for how to use virtual environment in conda.


3. Activate environment and conda install the cytnx package:

    Once you create a virtual environment, we need to activate the environment before starting to use it.

.. code-block:: shell

    $conda activate cytnx

.. code-block:: shell

    $conda install -c kaihsinwu cytnx

.. note::

    * To install the GPU (CUDA) support version, use:

    $conda install -c kaihsinwu cytnx_cuda


Once it is installed, we are all set, and ready to start using Cytnx.


Using Python API after Conda install
---------------------------------------
After installing Cytnx, using the Python API is very straight forward, simply import cytnx via:

* In Python:

.. code-block:: python
    :linenos:

    import cytnx

    A = cytnx.ones(4)
    print(A)

Output>>

.. code-block:: text

    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4)
    [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]

Using the C++ API
---------------------------------------
The conda package and the PyPI wheel install only the compiled Python extension; they no longer carry the C++ headers or a linkable ``libcytnx``. To build your own C++ code against Cytnx, build and install the C++ library directly with CMake and consume it via ``find_package(Cytnx)``, as described in :doc:`adv_install`.






.. toctree::
