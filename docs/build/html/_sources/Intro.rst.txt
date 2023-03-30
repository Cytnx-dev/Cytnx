Introduction
=================================
    Cytnx is a library designed for Quantum/classical Physics simulations. 

    The library is built from bottom-up, with both C++ and python in mind right at the beginning of development. That's why nearly 95% of the APIs are exactly the same at both C++ and python ends. 

    Most of Cytnx APIs share very similar interfaces as the most common and popular libraries: numpy/scipy/pytorch. This is specifically designed so as to reduce the learning curve for users. Furthermore, we implement these easy-to-use python libraries interfacing to the C++ side in hope to benefit users who want to bring their python programming experience to the C++ side and speed up their programs. 
    
    Cytnx also supports multi-devices (CPUs/GPUs) directly on the base container level. Especially, not only the container but also our linear algebra functions share the same APIs regardless of the devices where the input Tensors are stored, just like pytorch. This provides users the ability to accelerate the code without worrying too much about details of multi-device programming. 
    
    From the physics side, cytnx provides powerful tools such as UniTensor, Network, Bond, Symmetry etc. These objects are built on top of Tensor objects, specifically aiming to reduce the developing work of Tensor network algorithms by simplifying the user interfaces. 

    In this user guide, both python and C++ APIs will be discussed, provided side-by-side for users to better understand how to use Cytnx, and to understand the conversion in between the Python API and the C++ API. 
    

Features
--------------
* C++ and python are co-existing, there is no one first.
* 95% of API are the same in C++ and Python.
  This means that one can do a fast prototype in python, and directly convert to C++ with extremely minimal re-writing of the codebase. 
* GPUs/CPUs multi-device support. 
* Easy to use user-interface similar to numpy/scipy/pytorch. 
* Enhanced tools specifically designed for Quantum/classical Physics simulations.  




