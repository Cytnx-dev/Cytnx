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

### 1. All the Storage and Tensor have mulitple type support.
    Avaliable types are : (please refer to \link Type Type \endlink)

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
    * simple moving btwn CPU and GPU (see \link cytnx::Device Device \endlink and below)


## Objects:
    * Storage   [Python binded]
    * \link cytnx::Tensor Tensor \endlink   [Python binded]
    * \link cytnx::Bond Bond \endlink     [Python binded]
    * \link cytnx::Accessor Accessor \endlink [C++ only]
    * \link cytnx::Symmetry Symmetry \endlink [Python binded]
    * \link cytnx::UniTensor UniTensor \endlink [Python binded]
    * \link cytnx::Network Network \endlink [Python binded]

## linear algebra functions:

See \link cytnx::linalg cytnx::linalg \endlink for further details
	|func        |   inplace | CPU | GPU  | callby Tensor   | Tensor | UniTensor|
    |-------------|-----------|-----|------|-----------------|--------|-----------|
	Add | \link cytnx::Tensor::Add_( const T& rhs) Add_\endlink |✓   | ✓   | \link cytnx::Tensor::Add( const T& rhs) Add\endlink | \link cytnx::linalg::Add(const cytnx::Tensor& Lt, const cytnx::Tensor& Rt) Add\endlink | \link cytnx::linalg::Add( const cytnx::UniTensor& Lt, const cytnx::UniTensor& Rt ) Add\endlink
	Sub | \link cytnx::Tensor::Sub_( const T& rhs) Sub_\endlink |✓   | ✓   | \link cytnx::Tensor::Sub( const T& rhs) Sub\endlink | \link cytnx::linalg::Sub(const cytnx::Tensor& Lt, const cytnx::Tensor& Rt) Sub\endlink | \link cytnx::linalg::Sub( const cytnx::UniTensor& Lt, const cytnx::UniTensor& Rt ) Sub\endlink
	Mul | \link cytnx::Tensor::Mul_( const T& rhs) Mul_\endlink |✓   | ✓   | \link cytnx::Tensor::Mul( const T& rhs) Mul\endlink | \link cytnx::linalg::Mul(const cytnx::Tensor& Lt, const cytnx::Tensor& Rt) Mul\endlink | \link cytnx::linalg::Mul( const cytnx::UniTensor& Lt, const cytnx::UniTensor& Rt ) Mul\endlink
	Div | \link cytnx::Tensor::Div_( const T& rhs) Div_\endlink |✓   | ✓   | \link cytnx::Tensor::Div( const T& rhs) Div\endlink | \link cytnx::linalg::Div(const cytnx::Tensor& Lt, const cytnx::Tensor& Rt) Div\endlink | \link cytnx::linalg::Div( const cytnx::UniTensor& Lt, const cytnx::UniTensor& Rt ) Div\endlink
	Mod | x |✓   | ✓   | \link cytnx::Tensor::Mod(const T& rhs) Mod\endlink | \link cytnx::linalg::Mod(const cytnx::Tensor& Lt, const cytnx::Tensor& Rt) Mod\endlink | \link cytnx::linalg::Mod( const cytnx::UniTensor& Lt, const cytnx::UniTensor& Rt ) Mod\endlink
	Cpr | x |✓   | ✓   | \link cytnx::Tensor::Cpr( const T& rhs) Cpr\endlink | \link cytnx::linalg::Cpr(const cytnx::Tensor& Lt, const cytnx::Tensor& Rt) Cpr\endlink | x
    +,+=|   \link cytnx::Tensor::operator+=(const T& rc) +=\endlink    |✓   | ✓   | \link cytnx::Tensor::operator+=(const T& rc) +=\endlink | \link cytnx::operator+(const cytnx::Tensor& Lt, const cytnx::Tensor& Rt) +\endlink,\link cytnx::Tensor::operator+=(const T& rc) +=\endlink| \link cytnx::operator+(const cytnx::UniTensor& Lt, const cytnx::UniTensor& Rt) +\endlink,\link cytnx::UniTensor::operator+=(const cytnx::UniTensor& rhs) +=\endlink
    -,-=|   \link cytnx::Tensor::operator-=(const T& rc) -=\endlink    |✓   | ✓   | \link cytnx::Tensor::operator-=(const T& rc) -=\endlink | \link cytnx::operator-(const cytnx::Tensor& Lt, const cytnx::Tensor& Rt) -\endlink,\link cytnx::Tensor::operator-=(const T& rc) -=\endlink| \link cytnx::operator-(const cytnx::UniTensor& Lt, const cytnx::UniTensor& Rt) -\endlink,\link cytnx::UniTensor::operator-=(const cytnx::UniTensor& rhs) -=\endlink
    *,*=|   \link cytnx::Tensor::operator*=(const T& rc) *=\endlink    |✓   | ✓   | \link cytnx::Tensor::operator*=(const T& rc) *=\endlink | \link cytnx::operator*(const cytnx::Tensor& Lt, const cytnx::Tensor& Rt) *\endlink,\link cytnx::Tensor::operator*=(const T& rc) *=\endlink| \link cytnx::operator*(const cytnx::UniTensor& Lt, const cytnx::UniTensor& Rt) *\endlink,\link cytnx::UniTensor::operator*=(const cytnx::UniTensor& rhs) *=\endlink
    /,/=|   \link cytnx::Tensor::operator/=(const T& rc) /=\endlink    |✓   | ✓   | \link cytnx::Tensor::operator/=(const T& rc) /=\endlink | \link cytnx::operator/(const cytnx::Tensor& Lt, const cytnx::Tensor& Rt) /\endlink,\link cytnx::Tensor::operator/=(const T& rc) /=\endlink| \link cytnx::operator/(const cytnx::UniTensor& Lt, const cytnx::UniTensor& Rt) /\endlink,\link cytnx::UniTensor::operator/=(const cytnx::UniTensor& rhs) /=\endlink
	Svd |      x    |✓   | ✓   | \link cytnx::Tensor::Svd(const bool& is_UvT) const Svd\endlink|\link cytnx::linalg::Svd(const cytnx::Tensor & Tin, const bool & is_UvT) Svd\endlink|\link cytnx::linalg::Svd(const cytnx::UniTensor & Tin, const bool & is_UvT ) Svd\endlink
	Gesvd |      x    |✓   | ✓   | x |\link cytnx::linalg::Gesvd(const cytnx::Tensor & Tin, const bool & is_U, const bool& is_vT) Gesvd\endlink|\link cytnx::linalg::Gesvd(const cytnx::UniTensor & Tin, const bool & is_U, const bool& is_vT) Gesvd\endlink
	Svd_truncate |      x    |✓   | ✓   | x |\link cytnx::linalg::Svd_truncate(const cytnx::Tensor& Tin, const cytnx_uint64& keepdim, const double& err, const bool& is_UvT, const unsigned int& return_err, const cytnx_uint64& mindim) Svd_truncate\endlink|\link cytnx::linalg::Svd_truncate(const cytnx::UniTensor& Tin, const cytnx_uint64& keepdim, const double& err, const bool& is_UvT, const unsigned int& return_err, const cytnx_uint64& mindim) Svd_truncate\endlink
	Gesvd_truncate |      x    |✓   | ✓   | x |\link cytnx::linalg::Gesvd_truncate(const cytnx::Tensor& Tin, const cytnx_uint64& keepdim, const double& err, const bool& is_U, const bool& is_vT, const unsigned int& return_err, const cytnx_uint64& mindim) Gesvd_truncate\endlink|\link cytnx::linalg::Gesvd_truncate(const cytnx::UniTensor& Tin, const cytnx_uint64& keepdim, const double& err, const bool& is_U, const bool& is_vT, const unsigned int& return_err, const cytnx_uint64& mindim) Gesvd_truncate\endlink|
	InvM | \link cytnx::linalg::InvM_(cytnx::Tensor& Tin) InvM_ \endlink |✓   | ✓   | \link cytnx::Tensor::InvM()const InvM \endlink |\link cytnx::linalg::InvM(const cytnx::Tensor& Tin) InvM \endlink|\link cytnx::linalg::InvM(const cytnx::UniTensor& Tin) InvM \endlink
	Inv | \link cytnx::linalg::Inv_(cytnx::Tensor& Tin, const double& clip) Inv_ \endlink |✓   | ✓   | \link cytnx::Tensor::Inv(const double& clip)const Inv \endlink |\link cytnx::linalg::Inv(const cytnx::Tensor& Tin, const double& clip) Inv \endlink|x
	Conj | \link cytnx::linalg::Conj_(cytnx::Tensor& Tin) Conj_ \endlink |✓   | ✓   | \link cytnx::Tensor::Conj() Conj \endlink |\link cytnx::linalg::Conj(const cytnx::Tensor& Tin) Conj \endlink|\link cytnx::linalg::Conj(const cytnx::UniTensor& Tin) Conj \endlink
	Exp | \link cytnx::linalg::Exp_(cytnx::Tensor& Tin) Exp_ \endlink |✓   | ✓   | \link cytnx::Tensor::Exp() Exp \endlink |\link cytnx::linalg::Exp(const cytnx::Tensor& Tin) Exp \endlink|x
	Expf | \link cytnx::linalg::Expf_(cytnx::Tensor& Tin) Expf_ \endlink |✓   | ✓   | x |\link cytnx::linalg::Expf(const cytnx::Tensor& Tin) Expf \endlink|x
	Eigh | x  |✓   | ✓   | \link cytnx::Tensor::Eigh(const bool& is_V, const bool& row_v)const Eigh\endlink |\link cytnx::linalg::Eigh(const cytnx::Tensor& Tin, const bool& is_V, const bool& row_v) Eigh\endlink|\link cytnx::linalg::Eigh(const cytnx::UniTensor& Tin, const bool& is_V, const bool& row_v) Eigh\endlink
	ExpH | x  |✓   | ✓   | x |\link cytnx::linalg::ExpH(const cytnx::Tensor& Tin, const T& a, const T& b) ExpH \endlink|\link cytnx::linalg::ExpH(const cytnx::UniTensor& Tin, const T& a, const T& b) ExpH \endlink
	ExpM | x  |✓   | x   | x |\link cytnx::linalg::ExpM(const cytnx::Tensor& Tin, const T& a, const T& b) ExpM \endlink|\link cytnx::linalg::ExpM(const cytnx::UniTensor& Tin, const T& a, const T& b) ExpM \endlink
	Matmul | x |✓   | ✓   | x |\link cytnx::linalg::Matmul(const cytnx::Tensor& TL, const cytnx::Tensor& TR) Matmul \endlink|x
	Diag | x  |✓   | ✓   | x |\link cytnx::linalg::Diag(const cytnx::Tensor& Tin) Diag \endlink|x
	Tensordot | x |✓   | ✓   | x |\link cytnx::linalg::Tensordot(const cytnx::Tensor& Tl, const cytnx::Tensor& Tr, const std::vector<cytnx_uint64>& idxl, const std::vector<cytnx_uint64>& idxr, const bool& cacheL, const bool& cacheR) Tensordot \endlink|x
	Outer | x |✓   | ✓   | x |\link cytnx::linalg::Outer(const cytnx::Tensor& Tl, const cytnx::Tensor& Tr) Outer \endlink|x
	Vectordot | x |✓   | ✓   | x |\link cytnx::linalg::Vectordot(const cytnx::Tensor& Tl, const cytnx::Tensor& Tr, const bool& is_conj) Vectordot \endlink|x
	Tridiag | x |✓   | ✓   | x |\link cytnx::linalg::Tridiag(const cytnx::Tensor& Diag, const cytnx::Tensor& Sub_diag, const bool& is_V, const bool& is_row, bool throw_excp) Tridiag \endlink|x
	Kron | x |✓   | ✓   | x |\link cytnx::linalg::Kron(const cytnx::Tensor& Tl, const cytnx::Tensor& Tr, const bool& Tl_pad_left, const bool& Tr_pad_left) Kron \endlink|x
	Norm | x |✓   | ✓   | \link cytnx::Tensor::Norm() Norm\endlink |\link cytnx::linalg::Norm(const cytnx::Tensor& Tin) Norm \endlink|\link cytnx::linalg::Norm(const cytnx::UniTensor& Tin) Norm \endlink
	Dot | x |✓   | ✓   | x |\link cytnx::linalg::Dot(const cytnx::Tensor& Tl, const cytnx::Tensor& Tr) Dot \endlink|x
	Eig | x  |✓   | x   | x |\link cytnx::linalg::Eig(const cytnx::Tensor& Tin, const bool& is_V, const bool& row_v) Eig\endlink|\link cytnx::linalg::Eig(const cytnx::UniTensor& Tin, const bool& is_V, const bool& row_v) Eig\endlink
	Pow | \link cytnx::linalg::Pow_(cytnx::Tensor& Tin, const double& p) Pow_ \endlink |✓   | ✓   | \link cytnx::Tensor::Pow(const cytnx_double& p)const Pow \endlink |\link cytnx::linalg::Pow(const cytnx::Tensor& Tin, const double& p) Pow \endlink|\link cytnx::linalg::Pow(const cytnx::UniTensor& Tin, const double& p) Pow \endlink
	Abs | \link cytnx::linalg::Abs_(cytnx::Tensor& Tin) Abs_ \endlink |✓   | ✓   | \link cytnx::Tensor::Abs()const Abs \endlink |\link cytnx::linalg::Abs(const cytnx::Tensor& Tin) Abs \endlink|x
	Qr | x |✓   | ✓   | x |\link cytnx::linalg::Qr(const cytnx::Tensor& Tin, const bool& is_tau) Qr \endlink|\link cytnx::linalg::Qr(const cytnx::UniTensor& Tin, const bool& is_tau) Qr \endlink
	Qdr | x |✓   | x   | x |\link cytnx::linalg::Qdr(const cytnx::Tensor& Tin, const bool& is_tau) Qdr \endlink|\link cytnx::linalg::Qdr(const cytnx::UniTensor& Tin, const bool& is_tau) Qdr \endlink
	Det | x |✓   | ✓   | x |\link cytnx::linalg::Det(const cytnx::Tensor& Tin) Det \endlink|x
	Min | x |✓   | ✓   | \link cytnx::Tensor::Min()const Min\endlink |\link cytnx::linalg::Min(const cytnx::Tensor& Tn) Min\endlink|x
	Max | x |✓   | ✓   | \link cytnx::Tensor::Max()const Max\endlink |\link cytnx::linalg::Max(const cytnx::Tensor& Tn) Max\endlink|x
	Sum | x |✓   | ✓   | x |\link cytnx::linalg::Sum(const cytnx::Tensor& Tn) Sum\endlink|x
	Trace | x |✓   | x   | \link cytnx::Tensor::Trace(const cytnx_uint64& a, const cytnx_uint64& b)const Trace\endlink |\link cytnx::linalg::Trace(const cytnx::Tensor& Tn, const cytnx_uint64& axisA, const cytnx_uint64& axisB) Trace\endlink|\link cytnx::linalg::Trace(const cytnx::UniTensor& Tn, const std::string& a, const std::string& b) Trace \endlink
	Matmul_dg | x |✓   | ✓   | x |\link cytnx::linalg::Matmul_dg(const cytnx::Tensor& TL, const cytnx::Tensor& TR) Matmul_dg \endlink|x
	Tensordot_dg | x |✓   | x   | x |\link cytnx::linalg::Tensordot_dg(const cytnx::Tensor& Tl, const cytnx::Tensor& Tr, const std::vector<cytnx_uint64>& idxl, const std::vector<cytnx_uint64>& idxr, const bool& diag_L) Tensordot_dg \endlink|x
	Lstsq | x |✓   | x   | x |\link cytnx::linalg::Lstsq(const cytnx::Tensor &A, const cytnx::Tensor &b, const float &rcond) Lstsq\endlink|x
	Axpy | \link cytnx::linalg::Axpy_(const Scalar &a, const cytnx::Tensor &x, cytnx::Tensor &y) Axpy_ \endlink |✓   | x   | x |\link cytnx::linalg::Axpy(const Scalar &a, const cytnx::Tensor &x, const cytnx::Tensor &y) Axpy \endlink|x
	Ger | x |✓   | ✓   | x |\link cytnx::linalg::Ger(const cytnx::Tensor &x, const cytnx::Tensor &y, const Scalar &a) Ger\endlink|x
	Gemm | \link cytnx::linalg::Gemm_(const Scalar &a, const cytnx::Tensor &x, const cytnx::Tensor &y, const Scalar& b, cytnx::Tensor& c) Gemm_\endlink |✓   | ✓   | x |\link cytnx::linalg::Gemm(const Scalar &a, const cytnx::Tensor &x, const cytnx::Tensor &y) Gemm\endlink|x
	Gemm_Batch | x |✓   | ✓   | x |\link cytnx::linalg::Gemm_Batch(const std::vector< cytnx_int64 >& m_array, const std::vector< cytnx_int64 >& n_array, const std::vector< cytnx_int64 >& k_array, const std::vector< Scalar >& 	alpha_array, const std::vector< cytnx::Tensor >& a_tensors, const std::vector< cytnx::Tensor >& b_tensors, const std::vector< Scalar >& beta_array, std::vector< cytnx::Tensor >& c_tensors, const cytnx_int64 group_count, const std::vector< cytnx_int64 >& group_size ) Gemm_Batch\endlink|x

**iterative solver:**
	|func        | CPU | GPU  | Tensor | UniTensor|
    |------------|-----|------|--------|----------|
	|Lanczos |✓   | ✓   | \link cytnx::linalg::Lanczos(cytnx::LinOp *Hop, const cytnx::Tensor& Tin, const std::string method, const double &CvgCrit, const unsigned int &Maxiter, const cytnx_uint64 &k, const bool &is_V, const bool &is_row, const cytnx_uint32 &max_krydim, const bool &verbose) Lanczos\endlink|\link cytnx::linalg::Lanczos(cytnx::LinOp *Hop, const cytnx::UniTensor& Tin, const std::string method, const double &CvgCrit, const unsigned int &Maxiter, const cytnx_uint64 &k, const bool &is_V, const bool &is_row, const cytnx_uint32 &max_krydim, const bool &verbose) Lanczos\endlink
	|Lanczos_Exp |✓   | x   | x |\link cytnx::linalg::Lanczos_Exp(cytnx::LinOp *Hop, const cytnx::UniTensor& Tin, const Scalar& tau, const double &CvgCrit, const unsigned int &Maxiter, const bool &verbose) Lanczos_Exp\endlink
	|Arnoldi |✓   | x   | \link cytnx::linalg::Arnoldi(cytnx::LinOp *Hop, const cytnx::Tensor& Tin, const std::string which, const cytnx_uint64& maxiter, const cytnx_double &cvg_crit, const cytnx_uint64& k, const bool& is_V, const bool &verbose) Arnoli\endlink |\link cytnx::linalg::Arnoldi(cytnx::LinOp *Hop, const cytnx::UniTensor& Tin, const std::string which, const cytnx_uint64& maxiter, const cytnx_double &cvg_crit, const cytnx_uint64& k, const bool& is_V, const bool &verbose) Arnoli\endlink

## Container Generators
    Tensor: \link cytnx::zeros zeros()\endlink, \link cytnx::ones ones()\endlink, \link cytnx::arange arange()\endlink, \link cytnx::identity identity()\endlink, \link cytnx::eye eye()\endlink,

## Physics Category
    Tensor: \link cytnx::physics::spin spin(),\endlink  \link cytnx::physics::pauli pauli()\endlink

## Random
    See \link cytnx::random cytnx::random \endlink for further details

      func    | UniTensor | Tensor | Storage | CPU | GPU
    ----------|-----------|--------|---------|-----|-----
    ^normal   | x | \link cytnx::random::normal(const std::vector< cytnx_uint64 > &Nelem, const double &mean, const double &std, const int &device, const unsigned int &seed, const unsigned int &dtype) normal\endlink | x | ✓   |  ✓
    ^uniform  | x | \link cytnx::random::uniform(const std::vector< cytnx_uint64 > &Nelem, const double &low, const double &high, const int &device, const unsigned int &seed, const unsigned int &dtype) uniform\endlink | x | ✓   |  ✓
    *normal_   | \link cytnx::random::normal_(cytnx::UniTensor& Tin, const double& mean, const double& std, const unsigned int& seed) normal_ \endlink | \link cytnx::random::normal_(cytnx::Tensor& Tin, const double& mean, const double& std, const unsigned int& seed) normal_ \endlink  | \link cytnx::random::normal_(cytnx::Storage& Sin, const double& mean, const double& std, const unsigned int& seed) normal_ \endlink | ✓   |  ✓
    *uniform_   | \link cytnx::random::uniform_(cytnx::UniTensor& Tin, const double& low, const double& high, const unsigned int& seed) uniform_ \endlink | \link cytnx::random::uniform_(cytnx::Tensor& Tin, const double& low, const double& high, const unsigned int& seed) uniform_ \endlink  | \link cytnx::random::uniform_(cytnx::Storage& Sin, const double& low, const double& high, const unsigned int& seed) uniform_ \endlink | ✓   |  ✓

    `*` this is initializer

    `^` this is generator

    \note The difference of initializer and generator is that initializer is used to initialize the Tensor, and generator generates a new Tensor.

## conda install
    **[Currently Linux only]**

    without CUDA
    * python 3.6/3.7/3.8: conda install -c kaihsinwu cytnx

    with CUDA
    * python 3.6/3.7/3.8: conda install -c kaihsinwu cytnx_cuda

## Some snippets:

### Storage
    * Memory container with GPU/CPU support.
      Type conversions (type casting between Storages)
      and moving between devices easily possible.
    * Generic type object, the behavior is very similar to Python.
```{.cpp}

        Storage A(400,Type.Double);
        for(int i=0;i<400;i++)
            A.at<double>(i) = i;

        Storage B = A; // A and B share same memory, this is similar to Python

        Storage C = A.to(Device.cuda+0);

```

### Tensor
    * A tensor, API very similar to numpy and pytorch.
    * Simple moving btwn CPU and GPU:
```{.cpp}

        Tensor A({3,4},Type.Double,Device.cpu); // create tensor on CPU (default)
        Tensor B({3,4},Type.Double,Device.cuda+0); // create tensor on GPU with gpu-id=0


        Tensor C = B; // C and B share same memory.

        // move A to GPU
        Tensor D = A.to(Device.cuda+0);

        // inplace move A to GPU
        A.to_(Device.cuda+0);

```
    * Type conversion possible:
```{.cpp}

        Tensor A({3,4},Type.Double);
        Tensor B = A.astype(Type.Uint64); // cast double to uint64_t

```
    * Virtual swap and permute. All the permute and swap operations do not change the underlying memory immediately. Minimized cost of moving elements.
    * Use `contiguous()` when needed to actually move the memory layout.
```{.cpp}

        Tensor A({3,4,5,2},Type.Double);
        A.permute_(0,3,1,2); // this will not change the memory, only the shape info is changed.
        cout << A.is_contiguous() << endl; // false

        A.contiguous_(); // call contiguous() to actually move the memory.
        cout << A.is_contiguous() << endl; // this will be true!

```
    * Access single element using `.at`
```{.cpp}

        Tensor A({3,4,5},Type.Double);
        double val = A.at<double>(0,2,2);

```
    * Access elements similar to Python slices:
```{.cpp}

        typedef Accessor ac;
        Tensor A({3,4,5},Type.Double);
        Tensor out = A(0,":","1:4");
        // equivalent to Python: out = A[0,:,1:4]

```

### UniTensor
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


------------------------------
## Developers & Maintainers
Creator and Project manager | Affiliation     | Email
----------------------------|-----------------|---------
Kai-Hsin Wu                 |Boston Univ., USA|kaihsinwu@gmail.com
\n

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
