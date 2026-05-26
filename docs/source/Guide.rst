User Guide
=================================
To use the library, simply include/import cytnx.

* In Python, using import:

.. code-block:: python
    :linenos:

    import cytnx

* In C++, using the include header:

.. code-block:: c++
    :linenos:

    #include "cytnx.hpp";


.. Note::
    In C++, there is a namespace **cytnx**.

Aliases in Python modules and C++ namespaces can be used equivalently, for example if we want to alias cytnx as cy,

* In Python:

.. code-block:: python
    :linenos:

    import cytnx as cy

This is equivalent in C++ to:

* In C++:

.. code-block:: c++
    :linenos:

    #include "cytnx.hpp";
    namespace cy=cytnx;


**Now we are ready to start using cytnx!**

Overview
************************

This is a comprehensive guide to explain the usage and features of Cytnx. It covers fundamental concepts, basic structures, and high-level objects. While this guide can be read from start to end for a comprehensive understanding, most users will be interested in specific topics to start with:

Start reading
---------------

**Beginners** are advised to start with the following chapters:

:ref:`Conventions` explains the basic concepts of the library such as naming conventions and the unified C++/Python framework. This chapter is therefore essential for all users.

:ref:`Tensor notation` introduces the abstract objects that Cytnx builds on, tensors. A graphical notation is introduced for convenience.

:ref:`UniTensor` concerns the object that users will most frequently use: a 'UniTensor', which contains not only the tensor elements but also further metadata and methods to characterize and manipulate the tensor. Typical tensor network implementations will be based on this object.

:ref:`Tensor decomposition` explains how a tensor can be split into a product of several tensors by a singular value decompositon, eigenvalue decomposition, or QR decomposition.

:ref:`Iterative solver` concerns efficient iterative methods, for example for obtaining the smallest eigenvalue.

:ref:`Linear algebra` contains a list of further linear algebra functions that can be applied to a tensor.

Advanced topics
----------------

The remaining sections cover **specific topics** or the **backend** of Cytnx:

:ref:`Device` is the object that defines where the data is stored, and is important for parallel and distributed computing on GPUs and CPUs.

:ref:`Tensor` contains the data together with basic information about the indices, such as the shape of the tensor. This object is similar to numpy.arrays or torch.Tensor.

:ref:`Storage` concerns the basic memory access.

:ref:`Scalar` is an elementary object that refers to a number, which can have different data types.

:ref:`Common APIs` lists methods that several objects in Cytnx have in common.



Contents
************************

.. toctree::
    :maxdepth: 3
    :numbered:

    guide/conventions.rst
    guide/Device.rst
    guide/basic_obj/Tensor.rst
    guide/basic_obj/Storage.rst
    guide/basic_obj/Scalar.rst
    guide/TNotation.rst
    guide/uniten.rst
    guide/contraction.rst
    guide/decomposition.rst
    guide/itersol.rst
    guide/linalg.rst
    guide/common.rst
