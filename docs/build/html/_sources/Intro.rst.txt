Introdoction
=================================
    Cytnx is a library design for Quantum/classical Physics simulation. 

    The library is build from bottum-up, with both C++ and python in mind right at the beginning of developement. That's why nearly 95% of the API are exactly the same in both C++ and python ends. 

    Most of Cytnx APIs share very similar interface as the most popular libraries: numpy/scipy/pytorch, so as to reduce the learning curve for users. Further more, we bring this easy-to-use python libraries interface to C++ in hope to benefit users who want to bring their python programming experience to C++ side and speed up their programs. 
    
    Cytnx also support multi-devices (CPUs/GPUs) directly in the base container level. Furthermore, not only the container but also the linear algebra fucntionsshare the same API regadless of the devices, just like pytorch. This provides user ability to accelerate the code without knowing too much details about multi-devices programming. 
    
    From the physics side, cytnx_extension namespace/submodule provides powerful tools such as CyTensor, Network, Bond, Symmetry objects that are built on top of Tensor object, that significantly simplify the developing work of Tensor network algorithm. 

    In this user guide, both python and C++ APIs will both be discussed, provided side-by-side for user to better understand the convertion, and how easy it is to convert in between Python API and C++ API. 

    

Features
--------------
* C++ and python are co-exists, there is no one first.
* 95% of API are the same in C++ and Python.
  This means one can do a fast prototype in python, and directly convert to C++ with extremely minimal re-writing of codebase. 
* GPUs/CPUs multi-devices support. 
* Easy to use user-interface similar to numpy/scipy/pytorch. 
* Enhance tools specifically designs for Quantum/classical Physics simulation.  




