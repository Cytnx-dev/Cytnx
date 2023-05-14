Storage
==========
Storage is a low-level container which handles the allocation and freeing of memory, as well as transferring data between different devices. All elements within a Tensor are stored inside a Storage object.

Typically, users won't directly interact with this object, but there may be instances where it proves useful in C++.

.. Note::
    
    Unlike Tensor, the memory layout of Storage is always contiguous. 


In the following, let's see how to use it: 

.. toctree::
    :maxdepth: 1

    Storage_1_create.rst
    Storage_2_access.rst
    Storage_3_expand.rst
    Storage_4_vec.rst
    Storage_5_io.rst
