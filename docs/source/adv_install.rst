Advanced Install of Cytnx
============================

Build/Install Cytnx from source
*********************************
For advanced user who wish to build Cytnx from source, we provide cmake install.


Dependencies
-------------------
Cytnx requires the following minimum dependencies:

* cmake >=3.14
* git
* make
* Boost v1.53+ [check_deleted, atomicadd, intrusive_ptr]
* openblas (or mkl, see below)
* gcc v13+ (recommand latest or equivalent clang on Mac/Linux with C++20 support) (required -std=c++20)

In addition, you might want to install the following optional dependencies if you want Cytnx to compile with features like openmp, mkl and/or CUDA support.

[Openmp]

* openmp

[MKL]

* intel mkl

[CUDA]

* Nvidia CUDA toolkit >= 12.0 (12.x and 13.x supported)
* Nvidia cuDNN library
* Nvidia cuTENSOR library >= 2.0
* Nvidia cuQuantum library


[Python API]

* python >= 3.10
* pybind11 >= 3.0.0
* python-graphviz
* graphviz
* numpy
* beartype

There are two methods how you can set-up all the dependencies before starting the build process:

1. Using conda to install dependencies
2. Directly install dependencies one-by-one via system package manager

.. Note::

    We recommend using conda to handle all the dependencies (including compiling tools).
    This is the simplest way as conda automatically resolves the whole path of each dependency, allowing cmake to automatically capture those.

    How it works?

        >> The conda-forge channel includes not only the Python package but also other pre-compiled libraries/compilers.



**Option A. Using anaconda/conda to install dependencies**

1. Install anaconda/miniconda, setting the virtual environments

* For Linux/WSL:

.. code-block:: shell

    $conda config --add channels conda-forge
    $conda create --name cytnx _openmp_mutex=*=*_llvm
    $conda activate cytnx
    $conda upgrade --all


* For MacOS:

.. code-block:: shell

    $conda config --add channels conda-forge
    $conda create --name cytnx llvm-openmp
    $conda activate cytnx
    $conda upgrade --all

.. Note::

    The last line is updating all the libraries such that they are all dependent on the conda-forge channel.


2. Install the following dependencies:

.. code-block:: shell

    $conda install cmake make boost boost-cpp git compilers numpy openblas arpack pybind11 beartype arpack


.. Note::

    1. This installation includes the compilers/linalg libraries provided by conda-forge, so the installation of compilers on system side is not required.
    2. Some packages may not be required, or additional packages need to be installed, depending on the compiling options. See below for further information. If mkl shall be used instead of openblas, use the following dpenedencies:

        .. code-block:: shell

            $conda install cmake make boost boost-cpp git compilers numpy mkl mkl-include mkl-service arpack pybind11 libblas=*=*mkl beartype arpack

    3. After the installation, an automated test based on gtest and benchmark can be run. This option needs to be activated in the install script. In this case, gtest needs to be installed as well:


        .. code-block:: shell

            $conda install gtest benchmark


.. Hint::

    Trouble shooting:

        1. Make sure **conda-forge** channel has the top priority. This should be assured by running

            .. code-block:: shell

                $conda config --add channels conda-forge.

        2. Make sure that the conda channel priority is **flexible** or **strict**. This can be achieved by

            .. code-block:: shell

                $conda config --set channel_priority strict

            or changing *~/.condarc* accordingly. You can check if the packages are correctly installed from *conda-forge* by running *$conda list* and checking the **Channels** row.
        3. Make sure libblas=mkl (you can check using *$conda list | grep libblas*)


3. In addition, if you want to have GPU support (compile with -DUSE_CUDA=on), then additional packages need to be installed:

.. .. code-block:: shell

..     $conda install cudatoolkit cudatoolkit-dev

.. code-block:: shell

    $conda install -c nvidia cuda

If cutensor shall be used as well (compile option -DUSE_CUTENSOR=ON), then further install the following:

.. code-block:: shell

    $conda install -c nvidia cutensor

Similarly, cuqauantum (compile option -DUSE_CUTENSOR=ON), requires:

.. code-block:: shell

    $conda install -c nvidia cuquantum

.. important::

    **If you install cuTENSOR and/or cuQuantum from NVIDIA tarballs**
    (instead of conda), you must configure two separate paths -- one for the
    build and one for runtime:

    1. **Build time** -- point CMake at the extracted directories so the
       libraries are found during configuration:

       .. code-block:: shell

           $export CUTENSOR_ROOT=/path/to/libcutensor-<version>
           $export CUQUANTUM_ROOT=/path/to/cuquantum-<version>

    2. **Runtime** -- the dynamic loader must also find the shared libraries
       when you ``import cytnx`` (or run a C++ binary). Tarball libraries live
       outside the system loader's default search path, and unlike conda they
       are **not** registered with ``ldconfig``, so add their library
       directories to ``LD_LIBRARY_PATH``:

       .. code-block:: shell

           $export LD_LIBRARY_PATH=$CUTENSOR_ROOT/lib:$CUQUANTUM_ROOT/lib:$LD_LIBRARY_PATH

       For cuTENSOR 2.x tarballs the libraries sit directly under ``lib/``, so
       the path above is sufficient at both build and runtime. Older cuTENSOR
       1.x tarballs instead use a per-CUDA subdirectory (``lib/<cuda-major>``,
       e.g. ``lib/12``); on that legacy layout point ``LD_LIBRARY_PATH`` at the
       subdirectory. Cytnx's ``FindCUTENSOR`` searches both layouts at build
       time, so the runtime path matches whichever one was found. If
       ``LD_LIBRARY_PATH`` is not set
       up, importing/running Cytnx fails with ``error while loading shared
       libraries: libcutensor.so... cannot open shared object file``, or
       silently binds to a mismatched system copy of the library if one is
       present. Add these ``export`` lines to your shell profile (e.g.
       ``~/.bashrc``) to make them persistent.

**Option B. Install dependencies via system package manager**

You can also choose to install dependencies directly from the system package manager, but one needs to carefully resolve the dependency path for cmake to capture them successfully.


.. warning::

    For MacOS, a standard brew install openblas will not work since it lacks lapacke.h wrapper support.

    If you are using MacOS, please install intel mkl (free) instead.

    For the Python API, we recommend installing Python using anaconda or miniconda.



Compiling process
-------------------
Once you installed all the dependencies, it is time to start building the Cytnx source code.

**Option A. Using cmake preset**

We support the ``cmake-presets`` tool for building the library starting from version
v1.1.0. You can find the configuration file in ``CMakePresets.json``. For example,
if you choose the ``openblas-cpu`` preset, use the following command to build:


.. code-block:: shell

    $cmake --preset openblas-cpu
    $cmake --build --preset openblas-cpu
.. note::

   If you are using Visual Studio Code, you can also take advantage of the
   *CMake Tools* extension, which provides built-in support for selecting and
   running CMake presets directly from the editor.


.. _option-b-cmake-install:

**Option B. Using cmake install**

.. note::

    Use Option B when you need a bare CMake install of the **C++** library
    (e.g. to link other CMake projects against ``libcytnx`` from a system
    prefix). If you only want the **Python** API, install via pip instead
    -- see *Option C* below. ``cmake --install`` no longer copies the
    ``cytnx/`` Python sources into the install prefix; that path is
    handled by pip + scikit-build-core to avoid producing two competing
    copies of the package.

Please see the following steps for the standard cmake compiling process and all the compiling options:


1. Create a build directory:

.. code-block:: shell

    $mkdir build
    $cd build

2. Use cmake to automatically generate compiling files:

.. code-block:: shell

    $cmake [option] <Cytnx repo directory>

The following are the available compiling option flags that you can specify in **[option]**:

+------------------------+-------------------+------------------------------------+
|       options          | default           |          description               |
+------------------------+-------------------+------------------------------------+
| -DCMAKE_INSTALL_PREFIX | /usr/local/cytnx  | Install destination of the library |
+------------------------+-------------------+------------------------------------+
| -DBUILD_PYTHON         |   ON              | Compile and install Python API     |
+------------------------+-------------------+------------------------------------+
| -DUSE_MKL              |   OFF             | Compile Cytnx with intel MKL lib.  |
|                        |                   | If =off, default link to openblas  |
+------------------------+-------------------+------------------------------------+
| -DUSE_OMP              |   ON              | Compile with openmp acceleration   |
|                        |                   | If USE_MKL=on, USE_OMP is forced=on|
+------------------------+-------------------+------------------------------------+
| -DUSE_CUDA             |   OFF             | Compile with CUDA GPU support      |
+------------------------+-------------------+------------------------------------+
| -DUSE_HPTT             |   OFF             | Accelerate tensor transpose with   |
|                        |                   | hptt                               |
+------------------------+-------------------+------------------------------------+

Additional options for HPTT if -DUSE_HPTT=on:

+-------------------------+-------------------+------------------------------------+
|       options           | default           |          description               |
+-------------------------+-------------------+------------------------------------+
| -DHPTT_ENABLE_FINE_TUNE |  OFF              | HPTT optimized with native hardware|
+-------------------------+-------------------+------------------------------------+
| -DHPTT_ENABLE_AVX       |  OFF              | Compile HPTT with AVX instruction  |
+-------------------------+-------------------+------------------------------------+
| -DHPTT_ENABLE_ARM       |  OFF              | Compile HPTT with ARM arch.        |
+-------------------------+-------------------+------------------------------------+
| -DHPTT_ENABLE_IBM       |  OFF              | Compile HPTT with ppc64le arch     |
+-------------------------+-------------------+------------------------------------+


3. Compile the code:

    $make

4. Install to the target location:

    $make install


.. _option-c-pip-install:

**Option C. Install the Python API via pip (recommended for Python users)**

If you only need to use Cytnx from Python, the simplest path is to let
``pip`` drive the build through scikit-build-core. From the repository
root:

.. code-block:: shell

    $ pip install .

or, for development (rebuilds incrementally and lets ``import cytnx``
resolve back to the source tree):

.. code-block:: shell

    $ pip install --editable .

This installs ``cytnx`` into your active Python environment as a normal
package, so ``import cytnx`` works without any ``sys.path`` manipulation.
The same dependencies described above (compilers, BLAS/LAPACK, etc.)
still need to be present; ``pip`` handles only the Python side.



Using Python API after self-build install
-------------------------------------------

If you used **Option C** (``pip install .`` or ``pip install --editable .``),
Cytnx is already importable -- ``import cytnx`` just works inside the
Python environment where you ran ``pip``. Skip to the example below.

If you used **Option A** (``Install.sh``) or **Option B** (bare CMake
install), the install prefix contains only the compiled extension and
its support files, not the ``cytnx/`` Python sources, so you cannot
``import cytnx`` from there directly. Re-run ``pip install --editable .``
from the repository root to make the Python API importable; the
already-compiled artifacts will be reused.

Once ``cytnx`` is importable in your Python environment:

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


Using the C++ API after a self-build install
------------------------------------------------
Cytnx no longer exposes ``__cpp_include__``/``__cpp_lib__`` Python attributes to introspect the C++ headers and library paths -- pip and conda installs don't carry those files at all (see :ref:`Option C <option-c-pip-install>` above). To compile your own C++ code against Cytnx, use :ref:`Option B (bare CMake install) <option-b-cmake-install>` and consume the installed package the standard CMake way:

.. code-block:: cmake

    find_package(Cytnx CONFIG REQUIRED PATHS ${CYTNX_ROOT})
    target_link_libraries(your_target PRIVATE Cytnx::cytnx)

where ``CYTNX_ROOT`` is the ``-DCMAKE_INSTALL_PREFIX`` used for the :ref:`Option B <option-b-cmake-install>` install. ``tests/downstream_find_package/`` in the Cytnx repository is a minimal example of a project consuming Cytnx this way.


Build troubleshooting
*************************************

Choosing between CUDA 12 and CUDA 13 when both are installed
-------------------------------------------------------------------------------------

Cytnx supports both CUDA 12.x and 13.x. If more than one CUDA toolkit is present
(for example CUDA 12 installed via ``apt`` and CUDA 13 unpacked under
``/usr/local/cuda-13``), you must select the one to build against; otherwise
CMake picks up whichever ``nvcc`` it finds first on ``PATH``, which may not be
the one you intended. There are two independent things to pin -- the **build**
toolkit and the **runtime** libraries.

**1. Build time -- select the compiler.** ``enable_language(CUDA)`` finds the
compiler only from ``CMAKE_CUDA_COMPILER``, the ``CUDACXX`` environment
variable, or ``nvcc`` on ``PATH`` (it does *not* reuse the libraries found by
``find_package(CUDAToolkit)``). Point it at the toolkit you want -- the rest of
the toolkit (libraries, headers, version) is then derived from that ``nvcc``:

.. code-block:: shell

    # Use CUDA 13:
    $cmake -S . -B build -DUSE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13/bin/nvcc

    # ...or use CUDA 12:
    $cmake -S . -B build -DUSE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc

Equivalent alternatives are ``export CUDACXX=/usr/local/cuda-13/bin/nvcc`` or
putting the desired ``bin`` directory first on ``PATH``
(``export PATH=/usr/local/cuda-13/bin:$PATH``). For the ``pip`` build, pass it
through scikit-build-core:

.. code-block:: shell

    $pip install . --config-settings=cmake.args="-DUSE_CUDA=ON;-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13/bin/nvcc"

The configure output reports the resolved toolkit (``CUDA Version``,
``CUDA Toolkit bin dir``, ``CUDA compiler (nvcc)``) -- check these match the
version you intended. CMake caches the detected compiler, so if you previously
configured with the wrong one, delete the build directory before reconfiguring;
a stale ``CMakeCache.txt`` keeps the old choice.

**2. Runtime -- match the shared libraries.** Selecting the compiler does not
control which CUDA runtime libraries are loaded at run time. The dynamic loader
resolves ``libcudart.so.<major>`` (and cuTENSOR/cuQuantum, etc.) via
``LD_LIBRARY_PATH``, then the ``ldconfig`` cache. If a different major version is
registered with ``ldconfig`` (commonly the ``apt`` copy under
``/usr/lib/x86_64-linux-gnu``), it can be loaded instead of the toolkit you
built against, causing version-skew crashes. To force the matching runtime, put
its library directory first:

.. code-block:: shell

    $export LD_LIBRARY_PATH=/usr/local/cuda-13/lib64:$LD_LIBRARY_PATH

(Adjust the path to the toolkit you built against; append the cuTENSOR/cuQuantum
``lib`` directories too if you enabled them -- see the tarball note above.)

CUDA device link fails with ``elfLink linker library load error``
-------------------------------------------------------------------------------------

**Symptom.** A CUDA-enabled build (``-DUSE_CUDA=ON``) configures successfully,
but the CUDA *device link* step aborts with::

    nvlink fatal   : elfLink linker library load error

On non-Apple builds Cytnx turns on interprocedural optimization
(``CMAKE_INTERPROCEDURAL_OPTIMIZATION``) which, together with CUDA separable
compilation, enables CUDA *device* link-time optimization (``nvcc -dlto``). The
device link step then asks ``nvlink`` to load the NVVM library, and the error
above means it could not.

**Cause.** This is *not* caused by the empty ``libpthread.a`` / ``librt.a`` /
``libdl.a`` stub archives that glibc 2.34+ ships -- those are tolerated by
``nvlink``. It is a layout problem specific to the Debian/Ubuntu
``nvidia-cuda-toolkit`` apt package. That package installs ``libnvvm.so`` into
the multiarch directory ``/usr/lib/x86_64-linux-gnu/`` but does not place it
under the toolkit's ``lib64`` directory, which is where ``nvcc`` tells
``nvlink`` to look. ``nvcc`` passes ``-nvvmpath=/usr/lib/nvidia-cuda-toolkit``,
so ``nvlink`` tries to open ``/usr/lib/nvidia-cuda-toolkit/lib64/libnvvm.so``
and finds nothing. Regular (non-LTO) device linking does not load NVVM, which is
why the failure appears only once device LTO is enabled.

**Fix (recommended): use a complete CUDA toolkit.** Install CUDA from conda or
NVIDIA's official installer instead of the distribution's
``nvidia-cuda-toolkit`` package, and make sure its ``nvcc`` is first on
``PATH``:

.. code-block:: shell

    $conda install -c nvidia cuda

A toolkit laid out this way keeps ``libnvvm.so`` under ``nvvm/lib64`` where
``nvlink`` expects it, so device LTO works with no further action. This is also
the layout the CUDA build presets assume.

**Workaround: keep the apt package and add the missing path.** If you must build
against the distribution package, create the directory ``nvlink`` searches and
symlink the packaged ``libnvvm`` library into it:

.. code-block:: shell

    $libnvvm_src=$(ls -1 /usr/lib/x86_64-linux-gnu/libnvvm.so* | sort -V | tail -1)
    $sudo mkdir -p /usr/lib/nvidia-cuda-toolkit/lib64
    $sudo ln -s "$libnvvm_src" /usr/lib/nvidia-cuda-toolkit/lib64/libnvvm.so

The ``ls … | sort -V | tail -1`` picks the highest-version file present
(``libnvvm.so.4``, ``libnvvm.so.4.0.0``, or an unversioned ``libnvvm.so``),
so the command works regardless of whether the development symlink was installed.
Then re-run the build. On non-x86_64 hosts the multiarch directory differs;
locate the real library first with ``find /usr -name 'libnvvm.so*'`` and adjust
``libnvvm_src`` accordingly.

**Alternative: disable device LTO.** If changing the toolkit layout is not
practical, configure with ``-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF`` to skip
device LTO entirely:

.. code-block:: shell

    $cmake --preset openblas-cuda -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF

The build will complete without link-time optimizations, which trades a small
runtime performance cost for compatibility with the unmodified apt package.


Check Cytnx version
*************************************
The current version of the library can be printed by:

* In Python:

.. code-block:: python
    :linenos:

    print("Cytnx version = ", cytnx.__version__)

Generate API documentation
*************************************
An API documentation can be generated from the source code of Cytnx by using doxygen. The generated documentation is accessible online as the `API reference <api/index.html>`__. To create it locally, make sure that doxygen is installed:

.. code-block:: shell

    $conda install doxygen

Then, use doxygen in the Cytnx source code folder to generate the API documentation:

.. code-block:: shell

    $doxygen docs.doxygen

The documentation is created in the folder **docs/**. You can open **docs/html/index.html** in your browser to access it.

.. toctree::
