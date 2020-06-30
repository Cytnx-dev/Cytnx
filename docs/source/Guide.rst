User Guide
=================================
To use the library, simply include/import cytnx. 

* In Python, using import 

.. code-block:: python
    :linenos:

    import cytnx

* In C++, using include header

.. code-block:: c++
    :linenos:

    #include "cytnx.hpp";


.. Note::
    In C++, there is a namespace **cytnx**. 

There are equivalence of alias between python module and c++ namespace, for example if we want to alias cytnx as cy, 

In python :

.. code-block:: python
    :linenos:

    import cytnx as cy

This is equivalent in C++ as:

.. code-block:: c++
    :linenos:

    #include "cytnx.hpp";
    namespace cy=cytnx;
    

**Now we are ready to start using cytnx!**

Continue reading:

.. toctree::
    :maxdepth: 3
    :numbered:

    guide/behavior.rst
    guide/basic_obj/Tensor.rst
    guide/basic_obj/Storage.rst
    guide/linalg.rst
    guide/itersol.rst
    guide/cyx.rst
    guide/xlinalg.rst
