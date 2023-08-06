Creating a Storage
-------------------
The storage can be created similarly to a Tensor. Note that Storage does not have the concept of *shape*, and behaves basically just like a **vector** in C++.

To create a Storage, with dtype=Type.Double on the CPU: 

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Storage_1_create_create.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Storage_1_create_create.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Storage_1_create_create.out
    :language: text

.. Note::
    
    [Deprecated] Storage by itself only allocates memory (using malloc) without initializing its elements. 
    
    [v0.6.6+] Storage behaves like a vector and initializes all elements to zero. 

.. Tip::

    1. Use **Storage.set_zeros()** or **Storage.fill()** if you want to set all the elements to zero or some arbitrary numbers. 
    2. For complex type Storage, you can use **.real()** and **.imag()** to get the real part/imaginary part of the data. 


Type conversion
****************
Conversion between different data types is possible for a Storage. Just like Tensor, call **Storage.astype()** to convert between different data types. 

The available data types are the same as for a Tensor, see :ref:`Tensor with different dtype and device`. 

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Storage_1_create_astype.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Storage_1_create_astype.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Storage_1_create_astype.out
    :language: text

Transfer between devices
************************
We can also transfer the storage between different devices. Similar to Tensor, we can use **Storage.to()**. 

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Storage_1_create_to.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Storage_1_create_to.cpp
    :language: c++
    :linenos:

Output>>

.. code-block:: text
    
    cytnx device: CPU
    cytnx device: CUDA/GPU-id:0
    cytnx device: CUDA/GPU-id:0


.. Hint::

    1. Like for a Tensor, **.device_str()** returns the device string while **.device()** returns the device ID (cpu=-1).

    2. **.to()** returns a copy on the target device. Use **.to_()** instead to move the current instance to a target device. 


Get Storage of Tensor
**************************
Internally, the data of a Tensor is saved in a Storage. We can get the Storage of a Tensor by using **Tensor.storage()**. 

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Storage_1_create_get_storage.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Storage_1_create_get_storage.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Storage_1_create_get_storage.out
    :language: text

.. Note::

    The return value is a *reference* to the Tensor's internal storage. This implies that any modification to this Storage will modify the Tensor accordingly. 


**[Important]** For a Tensor in non-contiguous status, the meta-data is detached from its memory handled by storage. In this case, calling **Tensor.storage()** will return the current memory layout, not the ordering according to the Tensor indices in the meta-data. 

We demonstrate this using the Python API. The C++ API can be used in a similar way. 

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Storage_1_create_contiguous_check.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Storage_1_create_contiguous_check.out
    :language: text


.. toctree::
