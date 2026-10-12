Install & Usage of Cytnx
============================
Cytnx publishes prebuilt Python wheels, so installing the Python API does not
require a compiler or a CUDA toolkit. The recommended way to get one is
`Pixi <https://pixi.sh>`_, which creates a project-local environment from a
lock file so everyone working from the same project gets the same versions.

If you want to call Cytnx from C++, or you need a build option the wheels are
not compiled with, see `Using the C++ API`_ below.


Install with Pixi
********************
1. Install Pixi by following the `Pixi installation guide
   <https://pixi.sh/latest/installation/>`_.

2. Create a project, pin a Python version Cytnx has wheels for, and add Cytnx:

.. code-block:: shell

    pixi init my-project
    cd my-project
    pixi add "python=3.12"
    pixi add --pypi cytnx

3. Run Python inside that environment:

.. code-block:: shell

    pixi run python

.. important::

    Pin the Python version. A bare ``pixi add python`` takes the newest
    CPython conda-forge offers, which is already past the newest Python Cytnx
    builds wheels for; the resolve then fails, reporting that it needs
    ``cp315`` while Cytnx publishes only ``cp39`` through ``cp314``. Any
    version in `Supported platforms`_ works, and ``3.12`` is a safe default.

    Adding Cytnx before any Python at all fails differently, with
    ``No Python interpreter found in the dependencies``.

.. warning::

    The current release on PyPI, ``cytnx`` 1.1.1, installs but then fails on
    ``import cytnx`` with ``AttributeError: module 'importlib' has no attribute
    'util'``. The bug is fixed in the development branch but is not in a
    release yet.

    Until the next patch release, take Cytnx from the nightly wheel index,
    which is built from the development branch:

    .. code-block:: shell

        pixi add --pypi cytnx --index https://pypi.anaconda.org/cytnx-nightly-wheels/simple

    The nightly index carries every build, so this first resolve can take a
    few minutes. It is not stuck.

.. note::

    We do not publish a native Windows wheel at this stage. On Windows, please
    use WSL.


GPU (CUDA) support
---------------------------------------
The CUDA build is a separate package, ``cytnx-cuda``. It is currently
published only on the nightly index, and only for Linux on x86_64:

.. code-block:: shell

    pixi add --pypi cytnx-cuda --index https://pypi.anaconda.org/cytnx-nightly-wheels/simple

The wheel bundles the CUDA runtime libraries it needs, so no system CUDA
toolkit is required. An NVIDIA GPU and a sufficiently recent driver are, of
course, needed to run on the GPU.


Install with pip
********************
If you are not using Pixi, the wheels install with ``pip`` like any other
package. Use a virtual environment so the install does not touch your system
Python, and create it with a Python version from `Supported platforms`_:

.. code-block:: shell

    python3.12 -m venv cytnx-env
    source cytnx-env/bin/activate
    pip install cytnx

.. note::

    ``pip`` cannot be told to prefer one index over another: with
    ``--extra-index-url`` it considers every index together and picks the
    highest version, and a nightly such as ``1.1.1.dev202610071206`` sorts
    *below* the ``1.1.1`` release. So ``pip install --pre --extra-index-url
    ...`` quietly installs the release, not the nightly. To get a nightly with
    ``pip``, pin one exactly::

        pip install --extra-index-url https://pypi.anaconda.org/cytnx-nightly-wheels/simple \
            cytnx==1.1.1.devYYYYMMDDHHMM

    picking a version from the `nightly index
    <https://pypi.anaconda.org/cytnx-nightly-wheels/simple/cytnx/>`_. Pixi and
    ``uv`` both resolve a named index directly and need no such pin.


Supported platforms
********************
The released wheels cover CPython 3.9 through 3.14 on:

.. list-table::
    :header-rows: 1
    :widths: 30 35 35

    * - Platform
      - CPU (``cytnx``)
      - CUDA (``cytnx-cuda``)
    * - Linux x86_64
      - yes
      - yes (nightly only)
    * - Linux aarch64
      - yes
      - no
    * - macOS, Apple silicon
      - yes (macOS 14+)
      - no
    * - macOS, Intel
      - yes (macOS 15+)
      - no
    * - Windows
      - no, use WSL
      - no, use WSL

The Linux wheels require glibc 2.27 or newer; musl-based distributions such as
Alpine are covered by separate ``musllinux`` wheels. There is no source
distribution on PyPI, so a platform without a wheel has to build from source as
described in :doc:`adv_install`.


Checking the installation
---------------------------------------
Importing ``cytnx`` and creating a tensor is enough to confirm the install
works:

* In Python:

.. code-block:: python
    :linenos:

    import cytnx

    print(cytnx.__version__)
    A = cytnx.ones(4)
    print(A)

Output>>

.. code-block:: text

    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4)
    [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]

.. note::

    Run this from any directory other than a clone of the Cytnx source tree.
    Inside the source tree the local ``cytnx/`` directory shadows the installed
    package and the import fails.


Using the C++ API
---------------------------------------
The wheels install only the compiled Python extension; they do not carry the
C++ headers or a linkable ``libcytnx``. To build your own C++ code against
Cytnx, build and install the C++ library directly with CMake and consume it via
``find_package(Cytnx)``, as described in :doc:`adv_install`.
