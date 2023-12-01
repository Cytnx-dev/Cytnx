# Cytnx [![Build Status (GitHub Actions)](https://github.com/kaihsin/Cytnx/actions/workflows/ci-cmake_tests.yml/badge.svg?branch=master)](https://github.com/kaihsin/Cytnx/actions/workflows/ci-cmake_tests.yml) [![codecov](https://codecov.io/gh/Cytnx-dev/Cytnx/branch/master/graph/badge.svg?token=IHXTX7UI6O)](https://codecov.io/gh/Cytnx-dev/Cytnx) [![Coverity Scan Build Status](https://scan.coverity.com/projects/28835/badge.svg)](https://scan.coverity.com/projects/cytnx-dev-cytnx)
[![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx/badges/version.svg)](https://anaconda.org/kaihsinwu/cytnx) [![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx/badges/platforms.svg)](https://anaconda.org/kaihsinwu/cytnx)

![alt text](./Icons/Icon_small.png)

## Install
See The following user guide for installation instructions and an introduction to Cytnx:

[https://kaihsinwu.gitlab.io/Cytnx_doc/install.html](https://kaihsinwu.gitlab.io/Cytnx_doc/install.html)

## Intro slide
[Cytnx_v0.5.pdf (dated 07/25/2020)](https://drive.google.com/file/d/1vuc_fTbwkL5t52glzvJ0nNRLPZxj5en6/view?usp=sharing)

## News
    [v0.9.x]

    Implementation of new data structure for symmetric UniTensor, which differs from previous versions




## API Documentation:

[https://kaihsinwu.gitlab.io/cytnx_api/](https://kaihsinwu.gitlab.io/cytnx_api/)

## User Guide [under construction]:

[Cytnx User Guide](https://kaihsinwu.gitlab.io/Cytnx_doc/)



## Objects:
    * Storage   [binded]
    * Tensor    [binded]
    * Accessor  [C++ only]
    * Bond      [binded]
    * Symmetry  [binded]
    * CyTensor	[binded]
    * Network   [binded]

## Feature:

### Python x C++
    Benefit from both side.
    One can do simple prototype on Python side
    and easy transfer to C++ with small effort!


```c++
    // C++ version:
    #include "cytnx.hpp"
    cytnx::Tensor A({3,4,5},cytnx::Type.Double,cytnx::Device.cpu)
```


```python
    # Python version:
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

	A repository with the following examples will be released soon under the Cytnx organization on github:

    See example/ folder or documentation for how to use API
    See example/iTEBD folder for implementation on iTEBD algo.
    See example/DMRG folder for implementation on DMRG algo.
    See example/iDMRG folder for implementation on iDMRG algo.
    See example/HOTRG folder for implementation on HOTRG algo for classical system.
    See example/ED folder for implementation using LinOp & Lanczos.


## Avaliable linear-algebra function (Keep updating):

      func        |   inplace | CPU | GPU  | callby tn   | Tn | CyTn (xlinalg)
    --------------|-----------|-----|------|-------------|----|----------------
      Add         |   x       |  Y  |  Y   |    Y        | Y  |   Y
      Sub         |   x       |  Y  |  Y   |    Y        | Y  |   Y
      Mul         |   x       |  Y  |  Y   |    Y        | Y  |   Y
      Div         |   x       |  Y  |  Y   |    Y        | Y  |   Y
      Cpr         |   x       |  Y  |  Y   |    Y        | Y  |   x
    --------------|-----------|-----|------|-------------|----|----------------
      +,+=[tn]    |   x       |  Y  |  Y   |    Y (Add_) | Y  |   Y
      -,-=[tn]    |   x       |  Y  |  Y   |    Y (Sub_) | Y  |   Y
      *,*=[tn]    |   x       |  Y  |  Y   |    Y (Mul_) | Y  |   Y
      /,/=[tn]    |   x       |  Y  |  Y   |    Y (Div_) | Y  |   Y
      ==[tn]      |   x       |  Y  |  Y   |    Y (Cpr_) | Y  |   x
    --------------|-----------|-----|------|-------------|----|----------------
      Svd         |   x       |  Y  |  Y   |    Y        | Y  |   Y
     *Svd_truncate|   x       |  Y  |  Y   |    N        | Y  |   Y
      InvM        |   InvM_   |  Y  |  Y   |    Y        | Y  |   N
      Inv         |   Inv _   |  Y  |  Y   |    Y        | Y  |   N
      Conj        |   Conj_   |  Y  |  Y   |    Y        | Y  |   Y
    --------------|-----------|-----|------|-------------|----|----------------
      Exp         |   Exp_    |  Y  |  Y   |    Y        | Y  |   N
      Expf        |   Expf_   |  Y  |  Y   |    Y        | Y  |   N
      Eigh        |   x       |  Y  |  Y   |    Y        | Y  |   N
     *ExpH        |   x       |  Y  |  Y   |    N        | Y  |   Y
     *ExpM        |   x       |  Y  |  N   |    N        | Y  |   Y
    --------------|-----------|-----|------|-------------|----|----------------
      Matmul      |   x       |  Y  |  Y   |    N        | Y  |   N
      Diag        |   x       |  Y  |  Y   |    N        | Y  |   N
    *Tensordot    |   x       |  Y  |  Y   |    N        | Y  |   N
     Outer        |   x       |  Y  |  Y   |    N        | Y  |   N
     Vectordot    |   x       |  Y  | .Y   |    N        | Y  |   N
    --------------|-----------|-----|------|-------------|----|----------------
      Tridiag     |   x       |  Y  |  N   |    N        | Y  |   N
     Kron         |   x       |  Y  |  N   |    N        | Y  |   N
     Norm         |   x       |  Y  |  Y   |    Y        | Y  |   N
    *Dot          |   x       |  Y  |  Y   |    N        | Y  |   N
     Eig          |   x       |  Y  |  N   |    N        | Y  |   N
    --------------|-----------|-----|------|-------------|----|----------------
     Pow          |   Pow_    |  Y  |  Y   |    Y        | Y  |   Y
     Abs          |   Abs_    |  Y  |  N   |    Y        | Y  |   N
     Qr           |   x       |  Y  |  N   |    N        | Y  |   Y
     Qdr          |   x       |  Y  |  N   |    N        | Y  |   Y
     Det          |   x       |  Y  |  N   |    N        | Y  |   N
    --------------|-----------|-----|------|-------------|----|----------------
     Min          |   x       |  Y  |  N   |    Y        | Y  |   N
     Max          |   x       |  Y  |  N   |    Y        | Y  |   N
    *Trace        |   x       |  Y  |  N   |    Y        | Y  |   Y
     Mod          |   x       |  Y  |  Y   |    Y        | Y  |   Y
    Matmul_dg     |   x       |  Y  |  Y   |    N        | Y  |   N
    --------------|-----------|-----|------|-------------|----|----------------
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
      func          | Tn  | Stor | CPU | GPU
    -----------------------------------------
    *Make_normal()  |  Y  |  Y   | Y   |  Y
    *Make_uniform() |  Y  |  Y   | Y   |  N
    ^normal()       |  Y  |  x   | Y   |  Y
    ^uniform()      |  Y  |  x   | Y   |  N

    * this is initializer
    ^ this is generator

    [Note] The difference between initializer and generator is that the initializer is used to initialize the Tensor, and the generator creates a new Tensor.

## How to contribute & get in contact
    If you want to contribute to the development of the library, you are more than welocome. No matter if you want to dig deep into the technical details of the library, help improving the documentation and make the library more accessible to new users, or if you want to contribute to the project with high level algorithms - we are happy to keep improving Cytnx together.
	Also, if you have any questions or suggestions, feel free to reach out to us.

	You can contact us by:
    * Discord:
[https://discord.gg/dyhF7CCE9D](https://discord.gg/dyhF7CCE9D)

    * Creating an issue on github if you find a bug or have a suggestion:

[https://github.com/Cytnx-dev/Cytnx/issues](https://github.com/Cytnx-dev/Cytnx/issues)

    * Email, see below

## Developers & Maintainers

    [Creator and Project manager]
    Kai-Hsin Wu     (Boston Univ., USA) kaihsinwu@gmail.com

    Chang Teng Lin  (NTU, Taiwan): major maintainer and developer
    Ke Hsu          (NTU, Taiwan): major maintainer and developer
    Hao Ti          (NTU, Taiwan): documentation and linalg
    Ying-Jer Kao    (NTU, Taiwan): setuptool, cmake


## Contributors

    PoChung Chen     (NCHU, Taiwan)
    Chia-Min Chung   (NSYSU, Taiwan)
    Manuel Schneider (NYCU, Taiwan)
    Yen-Hsin Wu      (NTU, Taiwan)
    Po-Kwan Wu       (OSU, USA)
    Wen-Han Kao      (UMN, USA)
    Yu-Hsueh Chen    (NTU, Taiwan)


## References

    * example/DMRG:
[https://www.tensors.net/dmrg](https://www.tensors.net/dmrg)

    * hptt library:
[https://github.com/springer13/hptt](https://github.com/springer13/hptt)
