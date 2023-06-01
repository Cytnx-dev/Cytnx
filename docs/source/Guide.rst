User Guide
=================================
To use the library, simply include/import cytnx. 

* In Python, using import 

.. code-block:: python
    :linenos:

    import cytnx

* In C++, using the include header

.. code-block:: c++
    :linenos:

    #include "cytnx.hpp";


.. Note::
    In C++, there is a namespace **cytnx**. 

Aliases in Python modules and C++ namespaces can be used equivalently, for example if we want to alias cytnx as cy, 

In Python :

.. code-block:: python
    :linenos:

    import cytnx as cy

This is equivalent in C++ to:

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
    guide/Device.rst
    guide/basic_obj/Tensor.rst
    guide/basic_obj/Storage.rst
    guide/basic_obj/Scalar.rst
    guide/cyx.rst
    guide/net.rst
    guide/linalg.rst
    guide/itersol.rst
    guide/xlinalg.rst
