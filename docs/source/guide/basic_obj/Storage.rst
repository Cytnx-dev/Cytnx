Storage
==========
Storage is the low-level container that handles the memory allocate/free and transfer between different devices. Inside the Tensor, all the elements are store inside Storage object. 

Most of the case user will not directly use this object, but in some cases it could be useful in C++. 

.. Note::
    
    Unlike Tensor, The memory layout of Storage is always contiguous. 


In the following, let's see how to use it: 

.. toctree::
    :maxdepth: 1

    Storage_1_create.rst
    Storage_2_access.rst
    Storage_3_expand.rst
    Storage_4_vec.rst
    Storage_5_io.rst
