# Cytnx [![Build Status (GitHub Actions)](https://github.com/kaihsin/Cytnx/actions/workflows/ci-cmake_tests.yml/badge.svg?branch=master)](https://github.com/kaihsin/Cytnx/actions/workflows/ci-cmake_tests.yml) [![codecov](https://codecov.io/gh/Cytnx-dev/Cytnx/branch/master/graph/badge.svg?token=IHXTX7UI6O)](https://codecov.io/gh/Cytnx-dev/Cytnx) [![Coverity Scan Build Status](https://scan.coverity.com/projects/28835/badge.svg)](https://scan.coverity.com/projects/cytnx-dev-cytnx)
[![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx/badges/version.svg)](https://anaconda.org/kaihsinwu/cytnx) [![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx/badges/platforms.svg)](https://anaconda.org/kaihsinwu/cytnx)

![alt text](./Icons/Icon_small.png)

## What is Cytnx (pronounced as *sci-tens*)?

Cytnx is a tensor network library designed for quantum physics simulations using tensor network algorithms, offering the following features:

* Most of the APIs are identical in C++ and Python, enabling seamless transitions between the two for prototyping and production.
* Cytnx APIs share very similar interfaces with popular libraries such as NumPy, SciPy, and PyTorch, minimizing the learning curve for new users.
* We implement these easy-to-use Python libraries interfacing to the C++ side in hope to benefit users who want to bring their Python programming experience to the C++ side and speed up their programs.
* Cytnx supports multi-device operations (CPUs/GPUs) directly at the base container level. Both the containers and linear algebra functions share consistent APIs regardless of the devices on which the input tensors are stored, similar to PyTorch.
* For algorithms in physics, Cytnx provides powerful tools such as UniTensor, Network, Symmetry etc. These objects are built on top of Tensor objects, specifically aiming to reduce the developing work of Tensor network algorithms by simplifying the user interfaces.

**Intro slides**
>[Cytnx_v0.5.pdf (dated 07/25/2020)](https://drive.google.com/file/d/1vuc_fTbwkL5t52glzvJ0nNRLPZxj5en6/view?usp=sharing)

## News
	[v1.0.0]
	This is the release of the version v1.0.0, which is the stable version of the project.
 **See also**
[Release Note](misc_doc/version.log).

## API Documentation:

[https://kaihsinwu.gitlab.io/cytnx_api/](https://kaihsinwu.gitlab.io/cytnx_api/)

## User Guide [under construction]:

[Cytnx User Guide](https://kaihsinwu.gitlab.io/Cytnx_doc/)



## Objects:
* Storage   [Python binded]
* Tensor    [Python binded]
* Accessor  [C++ only]
* Bond      [Python binded]
* Symmetry  [Python binded]
* UniTensor [Python binded]
* Network   [Python binded]

## Features:

### Python & C++
>Benefit from both sides!
 One can do simple prototyping in Python
 and easy transfer code to C++ with small effort!


```c++
// C++ version:
#include "cytnx.hpp"
cytnx::Tensor A({3,4,5},cytnx::Type.Double,cytnx::Device.cpu);
```


```python
# Python version:
import cytnx
A =  cytnx.Tensor((3,4,5),dtype=cytnx.Type.Double,device=cytnx.Device.cpu)
```


### 1. All Storage and Tensor objects support multiple types.
Available types are :

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
          Type conversions (type casting between Storages)
          and moving between devices easily possible.
* Generic type object, the behavior is very similar to Python.

```c++
Storage A(400,Type.Double);
for(int i=0;i<400;i++)
  A.at<double>(i) = i;

Storage B = A; // A and B share same memory, this is similar to Python

Storage C = A.to(Device.cuda+0);
```


### 3. Tensor
* A tensor, API very similar to numpy and pytorch.
* Simple moving between CPU and GPU:

```c++
Tensor A({3,4},Type.Double,Device.cpu); // create tensor on CPU (default)
Tensor B({3,4},Type.Double,Device.cuda+0); // create tensor on GPU with gpu-id=0

Tensor C = B; // C and B share same memory.

// move A to GPU
Tensor D = A.to(Device.cuda+0);

// inplace move A to GPU
A.to_(Device.cuda+0);
```
* Type conversion possible:
```c++
Tensor A({3,4},Type.Double);
Tensor B = A.astype(Type.Uint64); // cast double to uint64_t
```

* Virtual swap and permute. All permute and swap operations do not change the underlying memory immediately. Minimizes cost of moving elements.
* Use `Contiguous()` when needed to actually move the memory layout.
```c++
Tensor A({3,4,5,2},Type.Double);
A.permute_(0,3,1,2); // this will not change the memory, only the shape info is changed.
cout << A.is_contiguous() << endl; // false

A.contiguous_(); // call Contiguous() to actually move the memory.
cout << A.is_contiguous() << endl; // true
```

* Access a single element using `.at`
```c++
Tensor A({3,4,5},Type.Double);
double val = A.at<double>(0,2,2);
```

* Access elements similar to Python slices:
```c++
typedef Accessor ac;
Tensor A({3,4,5},Type.Double);
Tensor out = A(0,":","1:4");
// equivalent to Python: out = A[0,:,1:4]

```

### 4. UniTensor
* Extension of Tensor, specifically designed for Tensor network simulations.
* `UniTensor` is a tensor with additional information such as `Bond`, `Symmetry` and `labels`. With these information, one can easily implement the tensor contraction.
```c++
Tensor A({3,4,5},Type.Double);
UniTensor tA = UniTensor(A); // convert directly.
UniTensor tB = UniTensor({Bond(3),Bond(4),Bond(5)},{}); // init from scratch.
// Relabel the tensor and then contract.
tA.relabels_({"common_1", "common_2", "out_a"});
tB.relabels_({"common_1", "common_2", "out_b"});
UniTensor out = cytnx::Contract(tA,tB);
tA.print_diagram();
tB.print_diagram();
out.print_diagram();
```
Output:
```
-----------------------
tensor Name :
tensor Rank : 3
block_form  : False
is_diag     : False
on device   : cytnx device: CPU
                 ---------
                /         \
   common_1 ____| 3     4 |____ common_2
                |         |
                |       5 |____ out_a
                \         /
                 ---------
-----------------------
tensor Name :
tensor Rank : 3
block_form  : False
is_diag     : False
on device   : cytnx device: CPU
                 ---------
                /         \
   common_1 ____| 3     4 |____ common_2
                |         |
                |       5 |____ out_b
                \         /
                 ---------
-----------------------
tensor Name :
tensor Rank : 2
block_form  : False
is_diag     : False
on device   : cytnx device: CPU
         --------
        /        \
        |      5 |____ out_a
        |        |
        |      5 |____ out_b
        \        /
         --------

```

* `UniTensor` supports `Block` form, which is useful if the physical system has a symmetry. See [user guide](https://kaihsinwu.gitlab.io/Cytnx_doc/) for more details.

## Linear Algebra
Cytnx provides a set of linear algebra functions.
* For instance, one can perform SVD, Eig, Eigh decomposition, etc. on a `Tensor` or `UniTensor`.
* Iterative methods such as Lanczos, Arnoldi are also available.
* The linear algebra functions are implemented in the `linalg` namespace.
For more details, see the [API documentation](https://kaihsinwu.gitlab.io/cytnx_api/).
```c++
auto mean = 0.0;
auto std = 1.0;
Tensor A = cytnx::random::normal({3, 4}, mean, std);
auto svds = cytnx::linalg::Svd(A); // SVD decomposition
```

## Examples
See the examples in the folder `example`

    See example/ folder or documentation for how to use API
    See example/iTEBD folder for implementation on iTEBD algo.
    See example/DMRG folder for implementation on DMRG algo.
    See example/TDVP folder for implementation on TDVP algo.
    See example/LinOp and example/ED folder for implementation using LinOp & Lanczos.


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
Creator and Project manager | Affiliation     | Email
----------------------------|-----------------|---------
Kai-Hsin Wu                 |Boston Univ., USA|kaihsinwu@gmail.com

Developers      | Affiliation     | Roles
----------------|-----------------|---------
Chang-Teng Lin  |NTU, Taiwan      |major maintainer and developer
Ke Hsu          |NTU, Taiwan      |major maintainer and developer
Ivana Gyro      |NTU, Taiwan      |major maintainer and developer
Hao-Ti Hung     |NTU, Taiwan      |documentation and linalg
Ying-Jer Kao    |NTU, Taiwan      |setuptool, cmake

## Contributors
Contributors    | Affiliation
----------------|-----------------
PoChung Chen    | NTHU, Taiwan
Chia-Min Chung  | NSYSU, Taiwan
Ian McCulloch   | NTHU, Taiwan
Manuel Schneider| NYCU, Taiwan
Yen-Hsin Wu     | NTU, Taiwan
Po-Kwan Wu      | OSU, USA
Wen-Han Kao     | UMN, USA
Yu-Hsueh Chen   | NTU, Taiwan
Yu-Cheng Lin    | NTU, Taiwan


## References
* Paper:
[https://arxiv.org/abs/2401.01921](https://arxiv.org/abs/2401.01921)

* Example/DMRG:
[https://www.tensors.net/dmrg](https://www.tensors.net/dmrg)

* hptt library:
[https://github.com/springer13/hptt](https://github.com/springer13/hptt)
