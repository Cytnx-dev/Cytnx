Introdoction
=================================
    Cytnx is a library design for Quantum/classical Physics simulation. 

    The library is build from bottum-up, with both C++ and python in mind right at the beginning of developement. That's why nearly 95% of the APIs are exactly the same in both C++ and python ends. 

    Most of Cytnx APIs share very similar interfaces as the most common and popular libraries: numpy/scipy/pytorch. This is spcifically designed so as to reduce the learning curve for users. Further more, we implement these easy-to-use python libraries interface to C++ side in hope to benefit users who want to bring their python programming experience to C++ side and speed up their programs. 
    
    Cytnx also support multi-devices (CPUs/GPUs) directly in the base container level. Especially, not only the container but also our linear algebra fucntions share the same APIs regadless of the devices where the input Tensors are, just like pytorch. This provides users ability to accelerate the code without worrying too much details about multi-devices programming. 
    
    From the physics side, cytnx_extension namespace/submodule provides powerful tools such as CyTensor, Network, Bond, Symmetry etc. These objects are built on top of Tensor objects, spcifically aiming for reduce the developing work of Tensor network algorithm by simplify the user interfaces. 

    In this user guide, both python and C++ APIs will be discussed, provided side-by-side for users to better understand how to use Cytnx, and understand the conversion in between Python API and C++ API. 
    

Features
--------------
* C++ and python are co-exists, there is no one first.
* 95% of API are the same in C++ and Python.
  This means one can do a fast prototype in python, and directly convert to C++ with extremely minimal re-writing of codebase. 
* GPUs/CPUs multi-devices support. 
* Easy to use user-interface similar to numpy/scipy/pytorch. 
* Enhance tools specifically designs for Quantum/classical Physics simulation.  




