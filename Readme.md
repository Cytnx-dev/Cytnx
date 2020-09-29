# Cytnx

![alt text](./Icon_small.png)


## Install 
See The following user guide for install and using of cytnx:

[https://kaihsinwu.gitlab.io/Cytnx_doc/install.html](https://kaihsinwu.gitlab.io/Cytnx_doc/install.html)

## Intro slide
[Cytnx_v0.5.pdf (dated 07/25/2020)](https://drive.google.com/file/d/1vuc_fTbwkL5t52glzvJ0nNRLPZxj5en6/view?usp=sharing)


[![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx/badges/version.svg)](https://anaconda.org/kaihsinwu/cytnx) [![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx/badges/platforms.svg)](https://anaconda.org/kaihsinwu/cytnx)
[![Build Status](https://travis-ci.org/kaihsin/Cytnx_build.svg?branch=master)](https://travis-ci.org/kaihsin/Cytnx_build)

## News
    [v0.6.2] 
 
## Stable Version:
[v0.6.2a](https://github.com/kaihsin/Cytnx/tree/v0.6.2a)

## Current dev Version:
    v0.6.3
    1. [Enhance] Add Device.Ncpus for detecting avaliable omp threads
    2. [Enhance] Add HPTT support on CPU permute.
    3. [Internal] Build version centralize
    4. [Enhance] More info for Device. 

    v0.6.2
    1. [Fix] Bug in CUDA Matmul interface passing the wrong object bug.  
    2. [Enhance] Add Matmul_dg for diagonal matrix mutiply dense matrix. 
    3. [Enhance] Add Tensordot_dg for tensordot with either Tl or Tr is diagonal matrix
    4. [Enhance] Contract dense & sparse memory optimized.      
    5. [example] Add iTEBD_gpu.py example
    6. [Fix] Bug in CUDA cuVectordot d and f seg fault 
    7. [Enhance] Add cuReduce for optimized reduction. 
    8. [Enhance] Optimize performance for Mul_internal_cpu. 
    9. [Enhance] Optimize performance for Div_internal_cpu. 
    10. [Fix] Bug in permute of UniTensor/Tensor with duplicate entries does not return error.

    v0.6.1
    1. [Enhance] add Scalar class (shadow)
    2. [Enhance] change default allocation from Malloc to Calloc.
    3. [Enhance] change storage.raw_ptr() to storage.data() and storage.data<>() 
    4. [Enhance] change storage.cap to STORAGE_DEFT_SZ that can be tune.
    5. [Enhance] adding Tproxy/Tproxy, Tproxy/Tensor, Tensor/Tproxy operation 
    6. [Enhance] Add mv_elem type for LinOp, which intrinsically omp the matvec operation.    
    7. [Fatal  ] Fix bug in Dot for Matrix-Vector multiplication on both GPU and CPU with complex&real float dtypes.

    v0.6.0
    1. [Enhance] behavior change the behavior of permute to prevent redundant copy in UniTensor and Tensor.
    2. add Tensor::same_data to check if two Tensor has same storage.
    3. [Enhance] the behavior of reshape in Tensor to prevent redundant copy.    
    4. [Enhance] behavior change all linalg to follow the same disipline for permute/reshape/contiguous
    5. [Enhance] add print() in C++ API    
    6. [Fix] reshape() does not share memory
    7. [Fix] BoolStorage print_elem does not show the first element in shape


## API Documentation:

[https://kaihsin.github.io/Cytnx/docs/html/index.html](https://kaihsin.github.io/Cytnx/docs/html/index.html)

## User Guide [under construction!]:

[Cytnx User Guide](https://kaihsinwu.gitlab.io/Cytnx_doc/)



## conda install [![Build Status](https://travis-ci.org/kaihsin/Cytnx_build.svg?branch=master)](https://travis-ci.org/kaihsin/Cytnx_build)


    [Note] For Windows user, please using WSL.

* Without CUDA            

    python 3.6/3.7/3.8: [![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx/badges/latest_release_date.svg)](https://anaconda.org/kaihsinwu/cytnx) [![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx/badges/platforms.svg)](https://anaconda.org/kaihsinwu/cytnx)

        conda install -c kaihsinwu cytnx

* with CUDA

    python 3.6/3.7/3.8: [![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx_cuda/badges/latest_release_date.svg)](https://anaconda.org/kaihsinwu/cytnx_cuda)

        conda install -c kaihsinwu cytnx_cuda
    



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
     Mod          |   x       |  Y  |  Y   |    Y        | Y  |   Y 
    Matmul_dg     |   x       |  Y  |  Y   |    N        | Y  |   N 
    --------------|-----------|-----|------|-------------|----|-------
    *Tensordot_dg |   x       |  Y  |  Y   |    N        | Y  |   N

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

    * hptt library:
        https://github.com/springer13/hptt


## Acknowledgement
    KHW whould like to thanks for the following contributor(s) for invaluable contribution to the library

    * PoChung Chen  (NCHU, Taiwan) : testing, and bug reporting


