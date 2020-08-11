# Cytnx

## Stable Version:
    [v0.5.5 pre-release](https://github.com/kaihsin/Cytnx/tree/v0.5.5a)
 

## News

## Current dev Version:
    v0.5.6
    1. [Enhance] change linalg::QR -> linalg::Qr for unify the function call 
    2. Fix bug in UniTensor Qr, R UniTensor labels bug.
    3. Add Qdr for UniTensor and Tensor.
    4. Fix minor bug in internal, Type.is_float for Uint32.
    5. [Enhance] accessor can now specify with vector. 
    6. [Enhance] Tproxy.item()
    7. Fix inplace reshape_() in new way templ. does not perform inplace operation

    v0.5.5a
    1. [Feature] Tensor can now using operator() to access elements just like python. 
    2. [Enhance] Access Tensor can now exactly the same using slice string as in python.
    3. [Enhance] at/reshape/permute in Tensor can now give args without braket{} as in python.
    4. [Enhance] Storage.Load to static, so it can match Tensor
    5. [Major] Remove cytnx_extension class, rename CyTensor->UniTensor 

## Feature:

### Python x C++
    Benefit from both side. 
    One can do simple prototype on python side 
    and easy transfer to C++ with small effort!


```{.cpp}

    // c++ version:
    #include "cytnx.hpp"
    cytnx::Tensor A({3,4,5},cytnx::Type.Double,cytnx::Device.cpu)


    
```


```{.py}

    # python version:
    import cytnx
    A =  cytnx.Tensor([3,4,5],dtype=cytnx.Type.Double,device=cytnx.Device.cpu)



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

### 2. Multiple devices support.
    * simple moving btwn CPU and GPU (see below)


## Objects:
    * \link cytnx::Storage Storage \endlink   [binded]
    * \link cytnx::Tensor Tensor \endlink   [binded]
    * \link cytnx::Bond Bond \endlink     [binded] 
    * \link cytnx::Accessor Accessor \endlink [c++ only]
    * \link cytnx::Symmetry Symmetry \endlink [binded]
    * \link cytnx::CyTensor CyTensor \endlink [binded]
    * \link cytnx::Network Network \endlink [binded]

## linear algebra functions: 
    See \link cytnx::linalg cytnx::linalg \endlink for further details

      func    |   inplace | CPU | GPU  | callby tn 
    ----------|-----------|-----|------|-----------
      \link cytnx::linalg::Add Add\endlink     |   x       |  Y  |  Y   |    Y
      \link cytnx::linalg::Sub Sub\endlink     |   x       |  Y  |  Y   |    Y
      \link cytnx::linalg::Mul Mul\endlink     |   x       |  Y  |  Y   |    Y
      \link cytnx::linalg::Div Div\endlink     |   x       |  Y  |  Y   |    Y
      \link cytnx::linalg::Cpr Cpr\endlink     |   x       |  Y  |  Y   |    Y
      +,+=[tn]|   x       |  Y  |  Y   |    Y (\link cytnx::Tensor::Add_ Tensor.Add_\endlink)
      -,-=[tn]|   x       |  Y  |  Y   |    Y (\link cytnx::Tensor::Sub_ Tensor.Sub_\endlink)
      *,*=[tn]|   x       |  Y  |  Y   |    Y (\link cytnx::Tensor::Mul_ Tensor.Mul_\endlink)
      /,/=[tn]|   x       |  Y  |  Y   |    Y (\link cytnx::Tensor::Div_ Tensor.Div_\endlink)
      ==  [tn]|   x       |  Y  |  Y   |    Y (\link cytnx::Tensor::Cpr_ Tensor.Cpr_\endlink)
      \link cytnx::linalg::Svd Svd\endlink     |   x       |  Y  |  Y   |    Y
      *\link cytnx::linalg::Svd_truncate Svd_truncate\endlink     |   x       |  Y  |  Y   |    N
      \link cytnx::linalg::InvM InvM\endlink     |   \link cytnx::linalg::InvM_ InvM_\endlink    |  Y  |  Y   |    Y
      \link cytnx::linalg::Inv Inv\endlink     |   \link cytnx::linalg::Inv_ Inv_\endlink    |  Y  |  Y   |    Y
      \link cytnx::linalg::Conj Conj\endlink    |   \link cytnx::linalg::Conj_ Conj_\endlink   |  Y  |  Y   |    Y 
      \link cytnx::linalg::Exp Exp\endlink     |   \link cytnx::linalg::Exp_ Exp_\endlink    |  Y  |  Y   |    Y
      \link cytnx::linalg::Expf Expf\endlink     |   \link cytnx::linalg::Expf_ Expf_\endlink    |  Y  |  Y   |    Y
      *\link cytnx::linalg::ExpH ExpH\endlink       |   x   |  Y  |  Y   |    N 
      *\link cytnx::linalg::ExpM ExpM\endlink       |   x   |  Y  |  Y   |    N 
      \link cytnx::linalg::Eigh Eigh\endlink    |   x       |  Y  |  Y   |    Y 
      \link cytnx::linalg::Matmul Matmul\endlink  |   x       |  Y  |  Y   |    N
      \link cytnx::linalg::Diag Diag\endlink    |   x       |  Y  |  Y   |    N
      *\link cytnx::linalg::Tensordot Tensordot\endlink | x | Y | Y | N
      \link cytnx::linalg::Outer Outer\endlink  |   x       | Y   | Y   |   N
      \link cytnx::linalg::Kron Kron\endlink  |   x       | Y   | N   |   N
      \link cytnx::linalg::Norm Norm\endlink  |   x       | Y   | Y   |   Y
      \link cytnx::linalg::Vectordot Vectordot\endlink    |   x       |  Y  | .Y   |    N
      \link cytnx::linalg::Tridiag Tridiag\endlink    | x | Y | N | N  
      *\link cytnx::linalg::Dot Dot\endlink | x | Y | Y | N
      \link cytnx::linalg::Eig Eig\endlink    |   x       |  Y  |  N   |    Y 
      \link cytnx::linalg::Pow Pow\endlink    |   \link cytnx::linalg::Pow_ Pow_\endlink       |  Y  |  Y   |    Y 
      \link cytnx::linalg::Abs Abs\endlink    |   \link cytnx::linalg::Abs_ Abs_\endlink       |  Y  |  N   |    Y
      \link cytnx::linalg::Qr Qr\endlink    |  x       |  Y  |  N   |    N
      \link cytnx::linalg::Qdr Qdr\endlink    |  x       |  Y  |  N   |    N
      \link cytnx::linalg::Min Min\endlink    |  x       |  Y  |  N   |    Y
      \link cytnx::linalg::Max Max\endlink    |  x       |  Y  |  N   |    Y
      *\link cytnx::linalg::Trace Trace\endlink | x | Y | N | N

    
    iterative solver
    
        \link cytnx::linalg::Lanczos_ER Lanczos_ER\endlink 
    

    * this is a high level linalg 

    ^ this is temporary disable

    . this is floating point type only

## Container Generators 
    Tensor: \link cytnx::zeros zeros()\endlink, \link cytnx::ones ones()\endlink, \link cytnx::arange arange()\endlink, \link cytnx::identity identity()\endlink, \link cytnx::eye eye()\endlink,

## Physics Category
    Tensor: \link cytnx::physics::spin spin()\endlink  \link cytnx::physics::pauli pauli()\endlink

## Random 
    See \link cytnx::random cytnx::random \endlink for further details

      func    | Tn  | Stor | CPU | GPU  
    ----------|-----|------|-----|-----------
    *\link cytnx::random::Make_normal Make_normal\endlink   |  Y  |  Y   | Y   |  Y
    ^\link cytnx::random::normal normal\endlink   |  Y  |  x   | Y   |  Y

    * this is initializer

    ^ this is generator

    [Note] The difference of initializer and generator is that initializer is used to initialize the Tensor, and generator generates a new Tensor.


    
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


## conda install  
    [Currently Linux only]

    without CUDA
    * python 3.6: conda install -c kaihsinwu cytnx_36
    * python 3.7: conda install -c kaihsinwu cytnx_37

    with CUDA
    * python 3.6: conda install -c kaihsinwu cytnx_cuda_36
    * python 3.7: conda install -c kaihsinwu cytnx_cuda_37

## compile
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

   
## Some snippets:

### Storage
    * Memory container with GPU/CPU support. 
      maintain type conversions (type casting btwn Storages) 
      and moving btwn devices.
    * Generic type object, the behavior is very similar to python.
```{.cpp}

        Storage A(400,Type.Double);
        for(int i=0;i<400;i++)
            A.at<double>(i) = i;

        Storage B = A; // A and B share same memory, this is similar as python 
        
        Storage C = A.to(Device.cuda+0); 

```

### Tensor
    * A tensor, API very similar to numpy and pytorch.
    * simple moving btwn CPU and GPU:
```{.cpp}

        Tensor A({3,4},Type.Double,Device.cpu); // create tensor on CPU (default)
        Tensor B({3,4},Type.Double,Device.cuda+0); // create tensor on GPU with gpu-id=0


        Tensor C = B; // C and B share same memory.

        // move A to gpu
        Tensor D = A.to(Device.cuda+0);

        // inplace move A to gpu
        A.to_(Device.cuda+0);

```
    * Type conversion in between avaliable:
```{.cpp}

        Tensor A({3,4},Type.Double);
        Tensor B = A.astype(Type.Uint64); // cast double to uint64_t

```
    * vitual swap and permute. All the permute and swap will not change the underlying memory
    * Use Contiguous() when needed to actual moving the memory layout.
```{.cpp}

        Tensor A({3,4,5,2},Type.Double);
        A.permute_(0,3,1,2); // this will not change the memory, only the shape info is changed.
        cout << A.is_contiguous() << endl; // this will be false!

        A.contiguous_(); // call Configuous() to actually move the memory.
        cout << A.is_contiguous() << endl; // this will be true!

```
    * access single element using .at
```{.cpp}

        Tensor A({3,4,5},Type.Double);
        double val = A.at<double>(0,2,2);

```
    * access elements with python slices similarity:
```{.cpp}

        typedef Accessor ac;
        Tensor A({3,4,5},Type.Double);
        Tensor out = A(0,":","1:4"); 
        // equivalent to python: out = A[0,:,1:4]    

```


     
## Fast Examples
    
    See test.cpp for using C++ .
    See test.py for using python  

## Developer

    Kai-Hsin Wu (Boston Univ.) kaihsinwu@gmail.com 

## Contributors

    Ying-Jer Kao (NTU, Taiwan): setuptool, cmake
    Yen-Hsin Wu (NTU, Taiwan): Network optimization
    Yu-Hsueh Chen (NTU, Taiwan): example, and testing
    Po-Kwan Wu (OSU): Icon optimization    
    Wen-Han Kao (UMN, USA) : testing of conda install 

## Refereces:

    * example/DMRG:
        https://www.tensors.net/dmrg


