Creating a UniTensor
--------------------

As mentioned in the introduction, a **UniTensor** consists of Block(s), Bond(s) and label(s). The Block(s) contain the data, while Bond(s) and label(s) are the meta data that describe the properties of the UniTensor.

.. image:: image/utcomp.png
    :width: 600
    :align: center




Generally, there are two types of UniTensor types: **un-tagged** and **tagged**, depending on whether the Bonds have a *direction*. In more advanced applications, a UniTensor may have block diagonal or other more complicated structure when symmetries are involved. Therefore, UniTensors can further be categorized into **non-symmetric** and **symmetric (block form)**:

+-----------+-----------------+-------------------------------+
|           |  non-symmetric  |  symmetric (block-diagonal)   |
+-----------+-----------------+-------------------------------+
| tagged    |     **O**       |            **O**              |
+-----------+-----------------+-------------------------------+
| untagged  |     **O**       |            **X**              |
+-----------+-----------------+-------------------------------+


In the following, we will explain how to construct a UniTensor.

Using generators
************************
Similar to the initialization of a Tensor, one can create a UniTensor through generators such as zero, ones, normal, uniform, arange and eye. The first argument provides shape information, which is used to construct the Bond objects and to determine the rank -- the number of tensor indices. Labels can be specified when creating a UniTensor, otherwise they are set to be "0", "1", "2", ... by default.

This gives us the first type of a UniTensor: an **untagged** UniTensor.

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_create_from_generator.py
    :language: python
    :linenos:

.. note::

    The generator **eye** expects the number of diagonal elements as a first argument instead of the shape of the resulting UniTensor.

.. note::

    The generator **arange** creates a one-dimensional UniTensor. In order to obtain a desired shape, use **reshape** and **set_rowrank** (see :ref:`reshape` and :ref:`rowrank`). If arange receives one argument, it is the number of elements. If three arguments are given, these correspond to start, stop, and stepsize. This syntax is similar to numpy.arange().

We can use **print_diagram()** to visualize a UniTensors in a more straightforward way as a diagram:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_create_from_generators_print_diagram.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_create_from_generators_print_diagram.out
    :language: text

The information provided by this output is explained in detail in :ref:`print_diagram()`.

Constructing from Tensor
************************

We can also convert a **cytnx.Tensor** into a UniTensor to create an **untagged** UniTensor.

In the following, we consider a simple rank-3 tensor as an example. The tensor diagram looks like:

.. image:: image/untag.png
    :width: 200
    :align: center

We can convert such a Tensor to a UniTensor:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_create_from_tensor.py
    :language: python
    :linenos:

Here, the Tensor **T** is converted to a UniTensor **uT** simply by wrapping it with constructor *cytnx.UniTensor()*. Formally, we can think of this as constructing a UniTensor **uT** with **T** being its *block* (data).
If we want to create a UniTensor with different dtype, for example, a complex UniTensor, we can do:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_create_complex.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_create_print_diagram.out
    :language: text

We see that a UniTensor with the same shape as *T* was created. The bond labels are set to the default values "0", "1" and "2".

From scratch
**************

Next, let's introduce the complete API for constructing a UniTensor from scratch:


.. py:function:: UniTensor(bonds, labels, rowrank, dtype, device, is_diag)

    :param List[cytnx.Bond] bonds: list of bonds
    :param List[string] labels: list of labels associate to each bond
    :param int rowrank: rowrank used when flattened into a matrix
    :param cytnx.Type dtype: the dtype of the block(s)
    :param cytnx.Device device: the device where the block(s) are held
    :param bool is_diag: whether the UniTensor is diagonal

The first argument **bonds** is a list of Bond objects. These correspond to the *shape* of a **cytnx.Tensor** where the elements in *shape* indicate the dimensions of the bonds. Here, each bond is represent by a **cytnx.Bond** object. In general, **cytnx.Bond** contains three things:

1. The dimension of the bond.
2. The direction of the bond (it can be BD_REG--undirectional, BD_KET (BD_IN)--inward, BD_BRA (BD_OUT)--outward)
3. The symmetry and the associate quantum numbers.

For more details, see :ref:`Bond`. Here, for simplicity, we will use only the dimension property of a Bond.

Now let's construct the rank-3 UniTensor with the same shape as in the above example. We assign the three bonds with labels ("a", "b", "c") and also set name to be "uT2 scratch".

.. image:: image/ut2.png
    :width: 300
    :align: center

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_create_scratch.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_create_scratch.out
    :language: text

.. note::

    The UniTensor will have all the elements in the block initialized with zeros.


Type conversion
**********************
It is possible to convert a UniTensor to a different data type. To convert the data type, simply use **UniTensor.astype()**.

For example, consider a UniTensor *A* with **dtype=Type.Int64**, which shall be converted to **Type.Double**:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_create_astype.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_create_astype.out
    :language: text


.. Note::

    UniTensor.dtype() returns a type-id, while UniTensor.dtype_str() returns the type name.
.. 1. A complex data type cannot directly be converted to a real data type. Use UniTensor.real() or UniTensor.imag() if you want to get the real or imaginary part.


Transfer between devices
***************************
Moving a UniTensor between different devices is very easy. We can use **UniTensor.to()** to move the UniTensor to a different device.

For example, let's create a UniTensor in the memory accessible by the CPU and transfer it to the GPU with gpu-id=0.

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_uniten_create_to.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_uniten_create_to.out
    :language: text

.. Note::

    1. You can use **UniTensor.device()** to get the current device-id (cpu = -1), whereas **UniTensor.device_str()** returns the device name.

    2. **UniTensor.to()** will return a copy on the target device. If you want to move the current Tensor to another device, use **UniTensor.to_()** (with underscore).


Tagged UniTensors and UniTensors with Symmetries
********************************************************

The creation of tagged, non-symmetric UniTensors will be explained in :ref:`Tagged UniTensor`. Symmetric UniTensors are discussed in :ref:`UniTensor with Symmetries`.
