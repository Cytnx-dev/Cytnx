# Cytnx

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
    * \link cytnx::UniTensor UniTensor \endlink [binded]
    * \link cytnx::Network Network \endlink
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
      \link cytnx::linalg::Inv Inv\endlink     |   \link cytnx::linalg::Inv_ Inv_\endlink    |  Y  |  Y   |    Y
      \link cytnx::linalg::Conj Conj\endlink    |   \link cytnx::linalg::Conj_ Conj_\endlink   |  Y  |  Y   |    Y 
      \link cytnx::linalg::Exp Exp\endlink     |   \link cytnx::linalg::Exp_ Exp_\endlink    |  Y  |  Y   |    Y
      \link cytnx::linalg::Eigh Eigh\endlink    |   x       |  Y  |  Y   |    Y 
      \link cytnx::linalg::Matmul Matmul\endlink  |   x       |  Y  |  Y   |    N
      \link cytnx::linalg::Diag Diag\endlink    |   x       |  Y  |  Y   |    N
      *\link cytnx::linalg::Tensordot Tensordot\endlink | x | Y | Y | N
      \link cytnx::linalg::Otimes Otimes\endlink  |   x       | Y   |  Y   |   N
 
    *this is a high level linalg 

## Generators 
    Tensor: \link cytnx::zeros zeros()\endlink, ones(), arange()
    
## Requirements
    * Boost v1.53+ [check_deleted, atomicadd, intrusive_ptr]
    * C++11
    * lapack 
    * blas 
    * gcc v6+

    [CUDA support]
    * CUDA v10+
    * cuDNN

    [OpenMp support]
    * openmp

    [Python]
    * pybind11 2.2.4
    * numpy >= 1.15 

## docker image with MKL 
  [https://hub.docker.com/r/kaihsinwu/cytnx_mkl](https://hub.docker.com/r/kaihsinwu/cytnx_mkl)
    
### To run:

```{.sh}
    $docker pull kaihsinwu/cytnx_mkl
    $docker run -ti kaihsinwu/cytnx_mkl
```

###Note:
    Once docker image is run, the user code can be compile (for example) with:

```{.sh}
    $g++-6 -std=c++11 -O3 <your.code.cpp> /opt/cytnx/libcytnx.so
```

## compile
    * compile: 

        $make -Bj4

    * turn on DEBUG mode:
        
        $make -Bj4 DEBUG_Enable=1

    * turn on OpenMp accelerate
        
        $make -Bj4 OMP_Enable=1 

    * turn on GPU accelerate
        
        $make -Bj4 GPU_Enable=1

    * turn on GPU+OpenMp accelerate
        
        $make -Bj4 GPU_Enable=1 OMP_Enable=1

    * compile python wrapper
        
        $make pyobj -Bj4 <args>

    <args> can be OMP_Enable, GPU_Enable, DEBUG_Enable, MKL_Enable.

    Note: if MKL_Enable=1, will enforce using icpc as compiler.
   
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
        A.permute_({0,3,1,2}); // this will not change the memory, only the shape info is changed.
        cout << A.is_contiguous() << endl; // this will be false!

        A.contiguous_(); // call Configuous() to actually move the memory.
        cout << A.is_contiguous() << endl; // this will be true!

```
    * access single element using .at
```{.cpp}

        Tensor A({3,4,5},Type.Double);
        double val = A.at<double>({0,2,2});

```
    * access elements with python slices similarity:
```{.cpp}

        typedef Accessor ac;
        Tensor A({3,4,5},Type.Double);
        Tensor out = A.get({ac(0),ac::all(),ac::range(1,4)}); 
        // equivalent to python: out = A[0,:,1:4]    

```


     
## Fast Examples
    
    See test.cpp for using C++ .
    See test.py for using python  

## Developer

    Kai-Hsin Wu kaihsinwu@gmail.com 




