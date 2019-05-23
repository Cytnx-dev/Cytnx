# TorX

## Requirements
    * Boost v1.53+ [check_deleted, atomicadd, intrusive_ptr]
    * C++11
    * lapack 
    * blas 
    * gcc v6+

    [CUDA support]
    * CUDA v10+
    * cuDNN
    
## ubuntu
    sudo apt-get install libboost-all-dev

## Objects:
    * Storage
    * Tensor
    * Bond

## Feature:

### Python x C++
    Benefit from both side. One can do simple prototype on python side and easy transfer to C++ with small effort!

### 1. All the Storage and Tensor can now have mulitple type support. 
        The avaliable types are :

        | tor10 type       | c++ type             | Type object
        |------------------|----------------------|--------------------
        | tor10_double     | double               | tor10type.Double
        | tor10_float      | float                | tor10type.Float
        | tor10_uint64     | uint64_t             | tor10type.Uint64
        | tor10_uint32     | uint32_t             | tor10type.Uint32
        | tor10_int64      | int64_t              | tor10type.Int64
        | tor10_int32      | int32_t              | tor10type.Int32
        | tor10_complex128 | std::complex<double> | tor10type.ComplexDouble
        | tor10_complex64  | std::complex<float>  | tor10type.ComplexFloat


### 2. Storage
        * Memory container with GPU/CPU support. maintain type conversions (type casting btwn Storages) and moving btwn devices.
        * Generic type object, the behavior is very similar to python.

```c++
            Storage A(400,tor10type.Double);
            for(int i=0;i<400;i++)
                A.at<double>(i) = i;

            Storage B = A; // A and B share same memory, this is similar as python 
            
            Storage C = A.to(tor10device.cuda+0); 
```


### 3. Tensor
        * A tensor, API very similar to numpy and pytorch.
        * simple moving btwn CPU and GPU:

```c++
            Tensor A({3,4},tor10type.Double,tor10device.cpu); // create tensor on CPU (default)
            Tensor B({3,4},tor10type.Double,tor10device.cuda+0); // create tensor on GPU with gpu-id=0


            Tensor C = B; // C and B share same memory.

            // move A to gpu
            Tensor D = A.to(tor10device.cuda+0);

            // inplace move A to gpu
            A.to_(tor10device.cuda+0);
```
        * Type conversion in between avaliable:
```c++
            Tensor A({3,4},tor10type.Double);
            Tensor B = A.astype(tor10type.Uint64); // cast double to uint64_t
```

        * vitual swap and permute. All the permute and swap will not change the underlying memory
        * Use Contiguous() when needed to actual moving the memory layout.
```c++
            Tensor A({3,4,5,2},tor10type.Double);
            A.permute({0,3,1,2}); // this will not change the memory, only the shape info is changed.
            cout << A.is_contiguous() << endl; // this will be false!

            A.Contiguous_(); // call Configuous() to actually move the memory.
            cout << A.is_contiguous() << endl; // this will be true!
```

## Avaliable linear-algebra function (Keep updating):

      func    |  inplace  | CPU | GPU
    ----------|-----------|-----|------
      Add     |   x       |  Y  |  Y
      Sub     |   x       |  Y  |  Y
      Mul     |   x       |  Y  |  Y
      Div     |   x       |  Y  |  Y
      +,+=[tn]|   x       |  Y  |  Y
      -,-=[tn]|   x       |  Y  |  Y
      *,*=[tn]|   x       |  Y  |  Y
      /,/=[tn]|   x       |  Y  |  Y

     
## Example
    
    See test.cpp for usage.

## Author

    Kai-Hsin Wu kaihsinwu@gmail.com 


