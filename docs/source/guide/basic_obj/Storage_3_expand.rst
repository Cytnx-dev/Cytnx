Increase size
-----------------
Just like C++ vectors, we can increase the size of the Storage.

append
********
It is possible to append a new element to the end of the Storage.
For example
* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Storage_3_expand_append.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Storage_3_expand_append.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Storage_3_expand_append.out
    :language: text

resize
********
Equivalently to *vector.resize* in C++, we can resize the Storage in Cytnx.

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Storage_3_expand_resize.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Storage_3_expand_resize.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Storage_3_expand_resize.out
    :language: text

.. Note::

    [Deprecated] If the size is increased in the resize operation, the additional elements will NOT be set to zero. Please use with care.

    [New][v0.6.6+] Additional elements are initialized by zeros when the memory is increased by resize. This behavior is similar to that of a vector.

.. Tip::

    1. You can use **Storage.size()** to get the current number of elements in the Storage.
    2. Internally, Cytnx allocates memory in multiples of 2. This optimizes the bandwidth use of CPU/GPU transfers and possibly increases the performance of some kernels. You can use **Storage.capacity()** to check the currently allocated number of elements in real memory, which might be larger than the number of elements in the Storage.


.. toctree::
