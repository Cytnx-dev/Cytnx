# Cytnx

![alt text](./Icon_small.png)

## Intro slide
[Cytnx_v0.5.pdf (dated 07/25/2020)](https://drive.google.com/file/d/1vuc_fTbwkL5t52glzvJ0nNRLPZxj5en6/view?usp=sharing)


[![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx_cuda_36/badges/version.svg)](https://anaconda.org/kaihsinwu/cytnx_cuda_36) [![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx_37/badges/platforms.svg)](https://anaconda.org/kaihsinwu/cytnx_37)
[![Build Status](https://travis-ci.com/kaihsin/Cytnx_build.svg?branch=master)](https://travis-ci.com/kaihsin/Cytnx_build)

## News
    [v0.5.5a] 

 
## Stable Version:
[v0.5.5a](https://github.com/kaihsin/Cytnx/tree/v0.5.5a)

## Current dev Version:
    v0.5.6
    1. [Enhance] change linalg::QR -> linalg::Qr for unify the function call 
    2. Fix bug in UniTensor Qr, R UniTensor labels bug.
    3. Add Qdr for UniTensor and Tensor.
    4. Fix minor bug in internal, Type.is_float for Uint32,Uint64.
    5. [Enhance] accessor can now specify with vector. 
    6. [Enhance] Tproxy.item()
    7. Fix inplace reshape_() in new way templ. does not perform inplace operation


    v0.5.5a
    1. [Feature] Tensor can now using operator() to access elements just like python. 
    2. [Enhance] Access Tensor can now exactly the same using slice string as in python.
    3. [Enhance] at/reshape/permute in Tensor can now give args without braket{} as in python.
    4. [Enhance] Storage.Load to static, so it can match Tensor
    5. [Major] Remove cytnx_extension class, rename CyTensor->UniTensor 
    6. Fix small bug in return ref of Tproxy 
    7. Fix bug in buffer size allocation in Svd_internal 
   


## API Documentation:

[https://kaihsin.github.io/Cytnx/docs/html/index.html](https://kaihsin.github.io/Cytnx/docs/html/index.html)

## User Guide [under construction!]:

[Cytnx User Guide](https://kaihsinwu.gitlab.io/Cytnx_doc/)

## conda install [![Build Status](https://travis-ci.com/kaihsin/Cytnx_build.svg?branch=master)](https://travis-ci.com/kaihsin/Cytnx_build)


    [Note] For Windows user, please using WSL.

* Without CUDA            

    python 3.6: [![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx_36/badges/latest_release_date.svg)](https://anaconda.org/kaihsinwu/cytnx_36) [![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx_36/badges/platforms.svg)](https://anaconda.org/kaihsinwu/cytnx_36)

        conda install -c kaihsinwu cytnx_36

    python 3.7: [![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx_37/badges/latest_release_date.svg)](https://anaconda.org/kaihsinwu/cytnx_37) [![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx_37/badges/platforms.svg)](https://anaconda.org/kaihsinwu/cytnx_37)

        conda install -c kaihsinwu cytnx_37   

* with CUDA

    python 3.6: [![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx_cuda_36/badges/latest_release_date.svg)](https://anaconda.org/kaihsinwu/cytnx_cuda_36)

        conda install -c kaihsinwu cytnx_cuda_36
    
    python 3.7: [![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx_cuda_37/badges/latest_release_date.svg)](https://anaconda.org/kaihsinwu/cytnx_cuda_37)

        conda install -c kaihsinwu cytnx_cuda_37  



## Requirements
    * Boost v1.53+ [check_deleted, atomicadd, intrusive_ptr]
    * C++11
    * lapack (lapacke or mkl) 
    * blas (or mkl) 
    * gcc v4.8.5+ (recommand v6+) (required -std=c++11) 

    [CUDA support]
    * CUDA v10+
    * cuDNN

    [OpenMp support]
    * openmp

    [Python]
    * pybind11 
    * numpy >= 1.15 

    [MKL]
    * icpc 
    

## ubuntu
    sudo apt-get install libboost-all-dev libopenblas-dev liblapack-dev liblapacke-dev cmake make curl g++ libomp-dev 


## MacOS
    1. brew install boost openblas lapack
    2. install anaconda, and conda install pybind11, numpy
    3. install intel mkl 

    For MacOS, there is no lapacke, so please use intel mkl instead. 

## Install 
    1.) create a build folder, and cd to the folder
        $mkdir build

        $cd build

    2.) resolving the dependency 

        $cmake [flags (optional)] <Cytnx repo folder>


        * -DUSE_ICPC (default = off)
            
            The default compiler is system's compiler. 
            
        * -DUSE_CUDA (default = off)

            If USE_CUDA=1, the code will compile with GPU support.

        * -DUSE_OMP (default = on)

            if USE_OMP=1, the code compiles with openmp accleration. 

        * -DUSE_MKL (default = off) [Recommend set it =on]

            If USE_MKL=off, the code will compile with the auto found blas/lapack vendors usually with blas_int=32bits. 
            If USE_MKL=on, the code will always compile with threaded mkl ILP64 library with blas_int=64bits 

            [Note] If you are not sure which version you are compile against to, use cytnx::__blasINTsize__/cytnx.__blasINTsize__ to check it's 64bits (=8) or 32bits (=4). 

        * -DCMAKE_INSTALL_PREFIX (default is /usr/local)
    
            Set the install target path.
        
    3.) compile by running:
       
        $make -Bj4

    4.) install to the target path.
        
        $make install


## Objects:
    * Storage   [binded]
    * Tensor    [binded]
    * Accessor  [c++ only]
    * Bond      [binded] 
    * Symmetry  [binded] 
    * CyTensor [binded] 
    * Network   [binded] 

## Feature:

### Python x C++
    Benefit from both side. 
    One can do simple prototype on python side 
    and easy transfer to C++ with small effort!


```c++
    // c++ version:
    #include "cytnx.hpp"
    cytnx::Tensor A({3,4,5},cytnx::Type.Double,cytnx::Device.cpu)
```


```python
    # python version:
    import cytnx
    A =  cytnx.Tensor((3,4,5),dtype=cytnx.Type.Double,device=cytnx.Device.cpu)
```


### 1. All the Storage and Tensor can now have mulitple type support. 
        The avaliable types are :

        | cytnx type       | c++ type             | Type object
        |------------------|----------------------|--------------------
        | cytnx_double     | double               | Type.Double
        | cytnx_float      | float                | Type.Float
        | cytnx_uint64     | uint64_t             | Type.Uint64
        | cytnx_uint32     | uint32_t             | Type.Uint32
        | cytnx_uint16     | uint16_t             | Type.Uint16
        | cytnx_int64      | int64_t              | Type.Int64
        | cytnx_int32      | int32_t              | Type.Int32
        | cytnx_int16      | int16_t              | Type.Int16
        | cytnx_complex128 | std::complex<double> | Type.ComplexDouble
        | cytnx_complex64  | std::complex<float>  | Type.ComplexFloat
        | cytnx_bool       | bool                 | Type.Bool

### 2. Storage
        * Memory container with GPU/CPU support. 
          maintain type conversions (type casting btwn Storages) 
          and moving btwn devices.
        * Generic type object, the behavior is very similar to python.

```c++
            Storage A(400,Type.Double);
            for(int i=0;i<400;i++)
                A.at<double>(i) = i;

            Storage B = A; // A and B share same memory, this is similar as python 
            
            Storage C = A.to(Device.cuda+0); 
```


### 3. Tensor
        * A tensor, API very similar to numpy and pytorch.
        * simple moving btwn CPU and GPU:

```c++
            Tensor A({3,4},Type.Double,Device.cpu); // create tensor on CPU (default)
            Tensor B({3,4},Type.Double,Device.cuda+0); // create tensor on GPU with gpu-id=0


            Tensor C = B; // C and B share same memory.

            // move A to gpu
            Tensor D = A.to(Device.cuda+0);

            // inplace move A to gpu
            A.to_(Device.cuda+0);
```
        * Type conversion in between avaliable:
```c++
            Tensor A({3,4},Type.Double);
            Tensor B = A.astype(Type.Uint64); // cast double to uint64_t
```

        * vitual swap and permute. All the permute and swap will not change the underlying memory
        * Use Contiguous() when needed to actual moving the memory layout.
```c++
            Tensor A({3,4,5,2},Type.Double);
            A.permute_(0,3,1,2); // this will not change the memory, only the shape info is changed.
            cout << A.is_contiguous() << endl; // this will be false!

            A.contiguous_(); // call Configuous() to actually move the memory.
            cout << A.is_contiguous() << endl; // this will be true!
```

        * access single element using .at
```c++
            Tensor A({3,4,5},Type.Double);
            double val = A.at<double>(0,2,2);
```

        * access elements with python slices similarity:
```c++
            typedef Accessor ac;
            Tensor A({3,4,5},Type.Double);
            Tensor out = A(0,":","1:4"); 
            // equivalent to python: out = A[0,:,1:4]
            
```

### 4. UniTensor
        * extension of Tensor, specifically design for Tensor network simulation. 

        * See Intro slide for more details
```c++
            Tensor A({3,4,5},Type.Double);
            UniTensor tA = UniTensor(A,2); // convert directly.

            UniTensor tB = UniTensor({Bond(3),Bond(4),Bond(5)},{},2); // init from scratch. 
```




## Examples
    
    See example/ folder or documentation for how to use API
    See example/iTEBD folder for implementation on iTEBD algo.
    See example/DMRG folder for implementation on DMRG algo.
    See example/iDMRG folder for implementation on iDMRG algo.
    See example/HOTRG folder for implementation on HOTRG algo for classical system.
    See example/ED folder for implementation using LinOp & Lanczos. 


## Avaliable linear-algebra function (Keep updating):

      func        |   inplace | CPU | GPU  | callby tn   | Tn | CyTn (xlinalg)
    --------------|-----------|-----|------|-------------|----|-------
      Add         |   x       |  Y  |  Y   |    Y        | Y  |   Y
      Sub         |   x       |  Y  |  Y   |    Y        | Y  |   Y
      Mul         |   x       |  Y  |  Y   |    Y        | Y  |   Y
      Div         |   x       |  Y  |  Y   |    Y        | Y  |   Y
      Cpr         |   x       |  Y  |  Y   |    Y        | Y  |   x
    --------------|-----------|-----|------|-------------|----|-------
      +,+=[tn]    |   x       |  Y  |  Y   |    Y (Add_) | Y  |   Y
      -,-=[tn]    |   x       |  Y  |  Y   |    Y (Sub_) | Y  |   Y
      *,*=[tn]    |   x       |  Y  |  Y   |    Y (Mul_) | Y  |   Y
      /,/=[tn]    |   x       |  Y  |  Y   |    Y (Div_) | Y  |   Y
      ==[tn]      |   x       |  Y  |  Y   |    Y (Cpr_) | Y  |   x 
    --------------|-----------|-----|------|-------------|----|-------
      Svd         |   x       |  Y  |  Y   |    Y        | Y  |   Y
     *Svd_truncate|   x       |  Y  |  Y   |    N        | Y  |   Y
      InvM        |   InvM_   |  Y  |  Y   |    Y        | Y  |   N
      Inv         |   Inv _   |  Y  |  Y   |    Y        | Y  |   N
      Conj        |   Conj_   |  Y  |  Y   |    Y        | Y  |   Y
    --------------|-----------|-----|------|-------------|----|-------
      Exp         |   Exp_    |  Y  |  Y   |    Y        | Y  |   N
      Expf        |   Expf_   |  Y  |  Y   |    Y        | Y  |   N
      Eigh        |   x       |  Y  |  Y   |    Y        | Y  |   N
     *ExpH        |   x       |  Y  |  Y   |    N        | Y  |   Y
     *ExpM        |   x       |  Y  |  N   |    N        | Y  |   Y
    --------------|-----------|-----|------|-------------|----|-------
      Matmul      |   x       |  Y  |  Y   |    N        | Y  |   N
      Diag        |   x       |  Y  |  Y   |    N        | Y  |   N
    *Tensordot    |   x       |  Y  |  Y   |    N        | Y  |   N
     Outer        |   x       |  Y  |  Y   |    N        | Y  |   N 
     Vectordot    |   x       |  Y  | .Y   |    N        | Y  |   N 
    --------------|-----------|-----|------|-------------|----|-------
      Tridiag     |   x       |  Y  |  N   |    N        | Y  |   N
     Kron         |   x       |  Y  |  N   |    N        | Y  |   N
     Norm         |   x       |  Y  |  Y   |    Y        | Y  |   N
    *Dot          |   x       |  Y  |  Y   |    N        | Y  |   N 
     Eig          |   x       |  Y  |  N   |    N        | Y  |   N 
    --------------|-----------|-----|------|-------------|----|-------
     Pow          |   Pow_    |  Y  |  Y   |    Y        | Y  |   Y 
     Abs          |   Abs_    |  Y  |  N   |    Y        | Y  |   N 
     Qr           |   x       |  Y  |  N   |    N        | Y  |   Y 
     Qdr          |   x       |  Y  |  N   |    N        | Y  |   Y 
     Det          |   x       |  Y  |  N   |    N        | Y  |   N
    --------------|-----------|-----|------|-------------|----|-------
     Min          |   x       |  Y  |  N   |    Y        | Y  |   N 
     Max          |   x       |  Y  |  N   |    Y        | Y  |   N 
    *Trace        |   x       |  Y  |  N   |    Y        | Y  |   Y


    iterative solver:
     
        Lanczos_ER           
    

    * this is a high level linalg 
    
    ^ this is temporary disable
    
    . this is floating point type only
 
## Container Generators 

    Tensor: zeros(), ones(), arange(), identity(), eye()

## Physics category 

    Tensor: pauli(), spin()
    
     
## Random 
      func        | Tn  | Stor | CPU | GPU  
    -----------------------------------------------------
    *Make_normal() |  Y  |  Y   | Y   |  Y
    *Make_uniform() |  Y  |  Y   | Y   |  N
    ^normal()      |  Y  |  x   | Y   |  Y
    ^uniform()      |  Y  |  x   | Y   |  N

    * this is initializer
    ^ this is generator

    [Note] The difference of initializer and generator is that initializer is used to initialize the Tensor, and generator generates a new Tensor.
     

## Developer

    Kai-Hsin Wu (Boston Univ.) kaihsinwu@gmail.com 


## Contributors

    Ying-Jer Kao (NTU, Taiwan): setuptool, cmake
    Yen-Hsin Wu (NTU, Taiwan): Network optimization
    Yu-Hsueh Chen (NTU, Taiwan): example, and testing
    Po-Kwan Wu (OSU): Icon optimization    
    Wen-Han Kao (UMN, USA) : testing of conda install 

## References

    * example/DMRG:
        https://www.tensors.net/dmrg

## Acknowledgement
    KHW whould like to thanks for the following contributor(s) for invaluable contribution to the library

    * PoChung Chen  (NCHU, Taiwan) : testing, and bug reporting


