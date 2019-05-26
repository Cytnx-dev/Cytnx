# Cytnx

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

## ubuntu
    sudo apt-get install libboost-all-dev


## compile
    * compile
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

    <args> can be OMP_Enable, GPU_Enable, DEBUG_Enable.
   

## Objects:
    * Storage
    * Tensor
    * Bond

## Feature:

### Python x C++
    Benefit from both side. One can do simple prototype on python side and easy transfer to C++ with small effort!

### 1. All the Storage and Tensor can now have mulitple type support. 
        The avaliable types are :

        | cytnx type       | c++ type             | Type object
        |------------------|----------------------|--------------------
        | cytnx_double     | double               | cytnxtype.Double
        | cytnx_float      | float                | cytnxtype.Float
        | cytnx_uint64     | uint64_t             | cytnxtype.Uint64
        | cytnx_uint32     | uint32_t             | cytnxtype.Uint32
        | cytnx_int64      | int64_t              | cytnxtype.Int64
        | cytnx_int32      | int32_t              | cytnxtype.Int32
        | cytnx_complex128 | std::complex<double> | cytnxtype.ComplexDouble
        | cytnx_complex64  | std::complex<float>  | cytnxtype.ComplexFloat


### 2. Storage
        * Memory container with GPU/CPU support. maintain type conversions (type casting btwn Storages) and moving btwn devices.
        * Generic type object, the behavior is very similar to python.

```c++
            Storage A(400,cytnxtype.Double);
            for(int i=0;i<400;i++)
                A.at<double>(i) = i;

            Storage B = A; // A and B share same memory, this is similar as python 
            
            Storage C = A.to(cytnxdevice.cuda+0); 
```


### 3. Tensor
        * A tensor, API very similar to numpy and pytorch.
        * simple moving btwn CPU and GPU:

```c++
            Tensor A({3,4},cytnxtype.Double,cytnxdevice.cpu); // create tensor on CPU (default)
            Tensor B({3,4},cytnxtype.Double,cytnxdevice.cuda+0); // create tensor on GPU with gpu-id=0


            Tensor C = B; // C and B share same memory.

            // move A to gpu
            Tensor D = A.to(cytnxdevice.cuda+0);

            // inplace move A to gpu
            A.to_(cytnxdevice.cuda+0);
```
        * Type conversion in between avaliable:
```c++
            Tensor A({3,4},cytnxtype.Double);
            Tensor B = A.astype(cytnxtype.Uint64); // cast double to uint64_t
```

        * vitual swap and permute. All the permute and swap will not change the underlying memory
        * Use Contiguous() when needed to actual moving the memory layout.
```c++
            Tensor A({3,4,5,2},cytnxtype.Double);
            A.permute({0,3,1,2}); // this will not change the memory, only the shape info is changed.
            cout << A.is_contiguous() << endl; // this will be false!

            A.Contiguous_(); // call Configuous() to actually move the memory.
            cout << A.is_contiguous() << endl; // this will be true!
```

## Avaliable linear-algebra function (Keep updating):

      func    |   inplace | CPU | GPU  | callby tn 
    ----------|-----------|-----|------|-----------
      Add     |   x       |  Y  |  Y   |    Y
      Sub     |   x       |  Y  |  Y   |    Y
      Mul     |   x       |  Y  |  Y   |    Y
      Div     |   x       |  Y  |  Y   |    Y
      +,+=[tn]|   x       |  Y  |  Y   |    Y (Add_)
      -,-=[tn]|   x       |  Y  |  Y   |    Y (Sub_)
      *,*=[tn]|   x       |  Y  |  Y   |    Y (Mul_)
      /,/=[tn]|   x       |  Y  |  Y   |    Y (Div_)
      Svd     |   x       |  Y  |  Y   |    Y
      Inv     |   Inv_    |  Y  |  Y   |    Y
      Conj    |   Conj_   |  Y  |  Y   |    Y 
      Exp     |   Exp_    |  Y  |  Y   |    Y
     
## Example
    
    See test.cpp for usage.

## Author

    Kai-Hsin Wu kaihsinwu@gmail.com 


