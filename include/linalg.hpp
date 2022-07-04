#ifndef _linalg_H_
#define _linalg_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include "Storage.hpp"
#include "UniTensor.hpp"
#include "Scalar.hpp"
#include "LinOp.hpp"
#include <functional>

namespace cytnx {
  // class Tensor;    //fwd
  // class UniTensor; //fwd
  // class LinOp;     //fwd

  namespace linalg {

    // Add:
    //==================================================
    /**
    @brief element-wise add
    */
    cytnx::UniTensor Add(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);
    template <class T>
    cytnx::UniTensor Add(const T &lc, const cytnx::UniTensor &Rt);
    template <class T>
    cytnx::UniTensor Add(const cytnx::UniTensor &Lt, const T &rc);

    // Sub:
    //==================================================
    /**
    @brief element-wise subtract
    */
    cytnx::UniTensor Sub(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);
    template <class T>
    cytnx::UniTensor Sub(const T &lc, const cytnx::UniTensor &Rt);
    template <class T>
    cytnx::UniTensor Sub(const cytnx::UniTensor &Lt, const T &rc);

    // Mul:
    //==================================================
    /**
    @brief element-wise subtract
    */
    cytnx::UniTensor Mul(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);
    template <class T>
    cytnx::UniTensor Mul(const T &lc, const cytnx::UniTensor &Rt);
    template <class T>
    cytnx::UniTensor Mul(const cytnx::UniTensor &Lt, const T &rc);

    // Div:
    //==================================================
    /**
    @brief element-wise divide
    */
    cytnx::UniTensor Div(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);
    template <class T>
    cytnx::UniTensor Div(const T &lc, const cytnx::UniTensor &Rt);
    template <class T>
    cytnx::UniTensor Div(const cytnx::UniTensor &Lt, const T &rc);

    // Mod:
    //==================================================
    /**
    @brief element-wise modulo
    */
    cytnx::UniTensor Mod(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);
    template <class T>
    cytnx::UniTensor Mod(const T &lc, const cytnx::UniTensor &Rt);
    template <class T>
    cytnx::UniTensor Mod(const cytnx::UniTensor &Lt, const T &rc);

    std::vector<cytnx::UniTensor> Svd(const cytnx::UniTensor &Tin, const bool &is_U = true,
                                      const bool &is_vT = true);
    std::vector<cytnx::UniTensor> Svd_truncate(const cytnx::UniTensor &Tin,
                                               const cytnx_uint64 &keepdim, const double &err = 0,
                                               const bool &is_U = true, const bool &is_vT = true,
                                               const bool &return_err = false);
    std::vector<cytnx::UniTensor> Hosvd(
      const cytnx::UniTensor &Tin, const std::vector<cytnx_uint64> &mode,
      const bool &is_core = true, const bool &is_Ls = false,
      const std::vector<cytnx_int64> &trucate_dim = std::vector<cytnx_int64>());

    cytnx::UniTensor ExpH(const cytnx::UniTensor &Tin, const double &a = 1, const double &b = 0);
    cytnx::UniTensor ExpM(const cytnx::UniTensor &Tin, const double &a = 1, const double &b = 0);
    cytnx::UniTensor Trace(const cytnx::UniTensor &Tin, const cytnx_int64 &a = 0,
                           const cytnx_int64 &b = 1, const bool &by_label = false);
    std::vector<cytnx::UniTensor> Qr(const cytnx::UniTensor &Tin, const bool &is_tau = false);
    std::vector<cytnx::UniTensor> Qdr(const cytnx::UniTensor &Tin, const bool &is_tau = false);

    // Pow:
    //==================================================
    /**
    @brief take power p on all the elements in UniTensor.
    @param p, the power
    @return
        [UniTensor]

    */
    UniTensor Pow(const UniTensor &Tin, const double &p);

    /**
    @brief inplace perform power on all the elements in UniTensor.
    @param Tin, the input UniTensor.
    @param p, the power.

    description:
        on return, the elements in Tin will be modified to it's exponetial value.
    */
    void Pow_(UniTensor &Tin, const double &p);

  }  // namespace linalg

  cytnx::UniTensor operator+(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);
  template <class T>
  cytnx::UniTensor operator+(const T &lc, const cytnx::UniTensor &Rt);
  template <class T>
  cytnx::UniTensor operator+(const cytnx::UniTensor &Lt, const T &rc);

  cytnx::UniTensor operator-(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);
  template <class T>
  cytnx::UniTensor operator-(const T &lc, const cytnx::UniTensor &Rt);
  template <class T>
  cytnx::UniTensor operator-(const cytnx::UniTensor &Lt, const T &rc);

  cytnx::UniTensor operator*(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);
  template <class T>
  cytnx::UniTensor operator*(const T &lc, const cytnx::UniTensor &Rt);
  template <class T>
  cytnx::UniTensor operator*(const cytnx::UniTensor &Lt, const T &rc);

  cytnx::UniTensor operator/(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);
  template <class T>
  cytnx::UniTensor operator/(const T &lc, const cytnx::UniTensor &Rt);
  template <class T>
  cytnx::UniTensor operator/(const cytnx::UniTensor &Lt, const T &rc);

  cytnx::UniTensor operator%(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);
  template <class T>
  cytnx::UniTensor operator%(const T &lc, const cytnx::UniTensor &Rt);
  template <class T>
  cytnx::UniTensor operator%(const cytnx::UniTensor &Lt, const T &rc);
}  // namespace cytnx

//====================================================================================
//====================================================================================
//====================================================================================
namespace cytnx {
  /**
  @namespace cytnx::linalg
  @brief linear algebra related functions.
  */
  namespace linalg {
    Tensor Add(const Tensor &Lt, const Tensor &Rt);
    template <class T>
    Tensor Add(const T &lc, const Tensor &Rt);
    template <class T>
    Tensor Add(const Tensor &Lt, const T &rc);

    void iAdd(Tensor &Lt, const Tensor &Rt);

    // Sub:
    //==================================================
    /**
    @brief element-wise subtract
    */
    Tensor Sub(const Tensor &Lt, const Tensor &Rt);
    template <class T>
    Tensor Sub(const T &lc, const Tensor &Rt);
    template <class T>
    Tensor Sub(const Tensor &Lt, const T &rc);

    void iSub(Tensor &Lt, const Tensor &Rt);

    // Mul:
    //==================================================
    /**
    @brief element-wise subtract
    */
    Tensor Mul(const Tensor &Lt, const Tensor &Rt);
    template <class T>
    Tensor Mul(const T &lc, const Tensor &Rt);
    template <class T>
    Tensor Mul(const Tensor &Lt, const T &rc);

    void iMul(Tensor &Lt, const Tensor &Rt);

    // Div:
    //==================================================
    /**
    @brief element-wise divide
    */
    Tensor Div(const Tensor &Lt, const Tensor &Rt);
    template <class T>
    Tensor Div(const T &lc, const Tensor &Rt);
    template <class T>
    Tensor Div(const Tensor &Lt, const T &rc);

    void iDiv(Tensor &Lt, const Tensor &Rt);

    // Mod:
    //==================================================
    /**
    @brief element-wise divide
    */
    Tensor Mod(const Tensor &Lt, const Tensor &Rt);
    template <class T>
    Tensor Mod(const T &lc, const Tensor &Rt);
    template <class T>
    Tensor Mod(const Tensor &Lt, const T &rc);

    // Cpr:
    //==================================================
    /**
    @brief element-wise compare
    */
    Tensor Cpr(const Tensor &Lt, const Tensor &Rt);
    template <class T>
    Tensor Cpr(const T &lc, const Tensor &Rt);
    template <class T>
    Tensor Cpr(const Tensor &Lt, const T &rc);

    // Norm:
    //=================================================
    /**
    @brief calculate the norm of a tensor.
    @param Tl input Tensor
    @return Tensor

    [Note]
        1. if the input tensor is rank-1, the frobenius norm is calculated.
        2. if the input tensor is rank-N with N>=2, the tensor will be flatten to 1d first, and
    calculate the frobenius norm.
    */
    Tensor Norm(const Tensor &Tl);

    // Det:
    //=================================================
    /**
    @brief calculate the determinant of a tensor.
    @param Tl input Tensor
    @return Tensor

    [Note]
        1. input tensor should be a NxN rank-2 Tensor.
    */
    Tensor Det(const Tensor &Tl);

    // Svd:
    //==================================================
    /**
    @brief Perform Singular-Value decomposition on a rank-2 Tensor.
    @param Tin a \link cytnx::Tensor Tensor \endlink, it should be a rank-2 tensor (matrix)
    @param is_U if return a left uniform matrix.
    @param is_vT if return a right uniform matrix.
    @return [std::vector<Tensors>]

        1. the first tensor is a 1-d tensor contanin the singular values
        2. the second tensor is the left uniform matrix [U], a 2-d tensor (matrix). It only return
    when is_U=true.
        3. the third tensor is the right uniform matrix [vT], a 2-d tensor (matrix). It only return
    when is_vT=true.
    */
    std::vector<Tensor> Svd(const Tensor &Tin, const bool &is_U = true, const bool &is_vT = true);

    // Svd_truncate:
    //==================================================
    std::vector<Tensor> Svd_truncate(const Tensor &Tin, const cytnx_uint64 &keepdim,
                                     const double &err = 0, const bool &is_U = true,
                                     const bool &is_vT = true, const bool &return_err = false);

    // Hosvd:
    std::vector<Tensor> Hosvd(
      const Tensor &Tin, const std::vector<cytnx_uint64> &mode, const bool &is_core = true,
      const bool &is_Ls = false,
      const std::vector<cytnx_int64> &trucate_dim = std::vector<cytnx_int64>());

    // Qr:
    //==================================================
    /**
    @brief Perform QR decomposition on a rank-2 Tensor.
    @param Tin a \link cytnx::Tensor Tensor \endlink, it should be a rank-2 tensor (matrix)
    @param is_tau if return the tau that contains the Householder reflectors that generate q along
    with r. The tau array contains scaling factors for the reflectors
    @return [std::vector<Tensors>]

        1. the first tensor is the orthomormal matrix [Q], a 2-d tensor (matrix)
        2. the second tensor is the right-upper triangular matrix [R], a 2-d tensor (matrix).
        3. the third tensor is the Householder reflectors [H], a 1-d tensor (matrix). It only return
    when is_tau=true.
    */
    std::vector<Tensor> Qr(const Tensor &Tin, const bool &is_tau = false);

    // Qdr:
    //==================================================
    /**
    @brief Perform QDR decomposition on a rank-2 Tensor.
    @param Tin a \link cytnx::Tensor Tensor \endlink, it should be a rank-2 tensor (matrix)
    @param is_tau if return the tau that contains the Householder reflectors that generate q along
    with r. The tau array contains scaling factors for the reflectors
    @return [std::vector<Tensors>]

        1. the first tensor is the orthomormal matrix [Q], a 2-d tensor (matrix)
        2. the second tensor is the diagonal matrix [D], a 1-d tensor (matrix).
        3. the third tensor is the right-upper triangular matrix [R], a 2-d tensor (matrix).
        4. the forth tensor is the Householder reflectors [H], a 1-d tensor (matrix). It only return
    when is_tau=true.
    */
    std::vector<Tensor> Qdr(const Tensor &Tin, const bool &is_tau = false);

    // Eigh:
    //==================================================
    /**
    @brief eigen-value decomposition for Hermitian matrix
    @param Tin The Tensor
    @param is_V return eigen vectors
    @param row_V if set to ture, the return eigen vectors will be row form.
    [Note] the Tin should be a rank-2 Tensor.
    */
    std::vector<Tensor> Eigh(const Tensor &Tin, const bool &is_V = true, const bool &row_v = false);

    // Eig:
    //==================================================
    /**
    @brief eigen-value decomposition for generic square matrix
    @param Tin The Tensor
    @param is_V return eigen vectors
    @param row_V if set to ture, the return eigen vectors will be row form.

    [Note] the Tin should be a rank-2 Tensor.
    */
    std::vector<Tensor> Eig(const Tensor &Tin, const bool &is_V = true, const bool &row_v = false);

    // Trace:
    //==================================================
    /**
    @brief perform trace over index.

    [Note] the Tn should be at-least rank-2 Tensor.
    */
    Tensor Trace(const Tensor &Tn, const cytnx_uint64 &axisA = 0, const cytnx_uint64 &axisB = 1);

    // Min:
    //==================================================
    /**
    @brief get the minimum element.

    [Note] For complex TN, only real part is compared.
    */
    Tensor Min(const Tensor &Tn);

    // Max:
    //==================================================
    /**
    @brief get the maximum element.

    [Note] For complex TN, only real part is compared.
    */
    Tensor Max(const Tensor &Tn);

    // Sum:
    //==================================================
    /**
    @brief get the sum of all the elements.

    */
    Tensor Sum(const Tensor &Tn);

    // Matmul:
    //==================================================
    /**
    @brief perform matrix multiplication on two tensors.

    [Note] the TL and TR should be both rank-2 Tensor.
    */
    Tensor Matmul(const Tensor &TL, const Tensor &TR);

    // Matmul:
    //==================================================
    /**
    @brief perform matrix multiplication on two Tensors with one rank-1 and the other rank-2 where
    the rank-1 represent the diagonal elements of the specific tensor.

    [Note] the TL and TR one of them should be rank-1 Tensor and the other should be rank-2 Tensor.
    */
    Tensor Matmul_dg(const Tensor &Tl, const Tensor &Tr);

    // InvM:
    //==================================================
    /**
    @brief Matrix inverse.
    @return
        [Tensor]

    [Note] the Tin should be a rank-2 Tensor.
    */
    Tensor InvM(const Tensor &Tin);
    /**
    @brief inplace perform Matrix inverse.

    description:
        on return, the Tin will be modified to it's inverse.

    [Note] the Tin should be a rank-2 Tensor.
    */
    void InvM_(Tensor &Tin);

    // Inv:
    //==================================================
    /**
    @brief Element-wise inverse with clip.
    @return
        [Tensor]

    description:
        Performs Elementwise inverse with clip. if A[i] < clip, then 1/A[i] = 0 will be set.

    [Note] For complex type Tensors, the square norm is used to determine the clip.

    */
    Tensor Inv(const Tensor &Tin, const double &clip);

    /**
    @brief inplace perform Element-wise inverse with clip.
    @return
        [Tensor]

    description:
        1. Performs Elementwise inverse with clip. if A[i] < clip, then 1/A[i] = 0 will be set.
        2. on return, all the elements will be modified to it's inverse. if Tin is integer type, it
    will automatically promote to Type.Double.

    [Note] For complex type Tensors, the square norm is used to determine the clip.

    */
    void Inv_(Tensor &Tin, const double &clip);

    // Conj:
    //==================================================
    /**
    @brief Conjugate all the element in Tensor.
    @return
        [Tensor]

    [Note]
        1. if the input Tensor is complex, then return a new Tensor with all the elements are
    conjugated.
        2. if the input Tensor is real, then return a copy of input Tensor.
    */
    Tensor Conj(const Tensor &Tin);
    /**
    @brief inplace perform Conjugate on all the element in Tensor.

    [Note]
        1. if the input Tensor is complex, the elements of input Tensor will all be conjugated.
        2. if the input Tensor is real, then nothing act.
    */
    void Conj_(Tensor &Tin);

    // Exp:
    //==================================================
    /**
    @brief Exponential all the element in Tensor.
    @return
        [Double Tensor] or [ComplexDouble Tensor]

    */
    Tensor Exp(const Tensor &Tin);

    /**
    @brief Exponential all the element in Tensor.
    @return
        [Float Tensor] or [ComplexFloat Tensor]

    */
    Tensor Expf(const Tensor &Tin);

    /**
    @brief inplace perform Exponential on all the element in Tensor.
    @param Tin, the input Tensor.

    description:
        1. on return, the elements in Tin will be modified to it's exponetial value.
        2. For Real, if the type is not Double, change the type of the input tensor to Double.
        3. For Complex, if input is ComplexFloat, promote to ComplexDouble.
    */
    void Exp_(Tensor &Tin);

    /**
    @brief inplace perform Exponential on all the element in Tensor.
    @param Tin, the input Tensor.

    description:
        1. on return, the elements in Tin will be modified to it's exponetial value.
        2. For Real, if the type is not Float, change the type of the input tensor to Float.
        3. For Complex, if input is ComplexDouble, promote to ComplexFloat.
    */
    void Expf_(Tensor &Tin);

    // Pow:
    //==================================================
    /**
    @brief take power p on all the elements in Tensor.
    @param p, the power
    @return
        [Tensor]

    */
    Tensor Pow(const Tensor &Tin, const double &p);

    /**
    @brief inplace perform power on all the elements in Tensor.
    @param Tin, the input Tensor.
    @param p, the power.

    description:
        on return, the elements in Tin will be modified to it's exponetial value.
    */
    void Pow_(Tensor &Tin, const double &p);

    // Abs:
    //==================================================
    /**
    @brief Elementwise absolute value.
    @param Tin tensor.
    @return
        [Tensor]

    */
    Tensor Abs(const Tensor &Tin);

    /**
    @brief inplace perform elementwiase absolute value.
    @param Tin, the input Tensor.

    description:
        on return, the elements in Tin will be modified to it's absolute value. Note that if the
    input tensor is complex, it will be modified to real type.
    */
    void Abs_(Tensor &Tin);

    // Diag:
    //==================================================
    /**
    @brief return a diagonal tensor with diagonal elements provided as Tin.
    @return
        [Tensor]

    description:
        the return Tensor will be rank-2, with shape=(L, L); where L is the number of elements in
    Tin.


    [Note] Tin should be a rank-1 Tensor.

    */
    Tensor Diag(const Tensor &Tin);

    // Tensordot:
    //==================================================
    /**
    @brief perform tensor dot by sum out the indices assigned of two Tensors.
    @param Tl Tensor #1
    @param Tr Tensor #2
    @param idxl the indices of rank of Tensor #1 that is going to sum with Tensor #2
    @param idxr the indices of rank of Tensor #2 that is going to sum with Tensor #1
    @param cacheL cache Tensor #1 (See user-guide for details)
    @param cacheR cache Tensor #2 (See user-guide for details)
    @return
        [Tensor]

    [Note]
        1. the elements in idxl and idxr have one to one correspondence.
        2. two tensors should on same device.
    */
    Tensor Tensordot(const Tensor &Tl, const Tensor &Tr, const std::vector<cytnx_uint64> &idxl,
                     const std::vector<cytnx_uint64> &idxr, const bool &cacheL = false,
                     const bool &cacheR = false);

    // Tensordot_dg:
    //==================================================
    /**
    @brief perform tensor dot by sum out the indices assigned of two Tensors, with either one of
    them to be a rank-2 diagonal tensor represented by a rank-2 tensor.
    @param Tl Tensor #1
    @param Tr Tensor #2
    @param idxl the indices of rank of Tensor #1 that is going to sum with Tensor #2
    @param idxr the indices of rank of Tensor #2 that is going to sum with Tensor #1
    @param diag_L if Tl(true)/Tr(false) is a diagnal matrix, represented by a rank-1 tensor.
    @return
        [Tensor]

    [Note]
        1. the elements in idxl and idxr have one to one correspondence.
        2. two tensors should on same device.
        3. if diag_L=true, Tl should be a rank-1 tensor as the diagonal elements of a diagonal
    matrix. if false, Tr should be a rank-1 tensor
    */
    Tensor Tensordot_dg(const Tensor &Tl, const Tensor &Tr, const std::vector<cytnx_uint64> &idxl,
                        const std::vector<cytnx_uint64> &idxr, const bool &diag_L);

    // Outer:
    //==================================================
    /**
    @brief perform outer produces of two rank-1 Tensor.
    @param Tl rank-1 Tensor #1
    @param Tr rank-1 Tensor #2
    @return
        [Tensor]

    description:
        if the Tensor #1 has [shape_1], and Tensor #2 has [shape_2]; then the return Tensor will
    have shape: concate(shape_1,shape_2)

    [Note]
        two tensor should on same device.

    */
    Tensor Outer(const Tensor &Tl, const Tensor &Tr);

    // Kron:
    //==================================================
    /**
    @brief perform kronecker produces of two Tensor.
    @param Tl rank-n Tensor #1
    @param Tr rank-m Tensor #2
    @param Tl_pad_left The padding scheme for Tl if Tl.rank != Tr.rank
    @param Tr_pad_left The padding scheme for Tr if Tl.rank != Tr.rank
    @return
        [Tensor]

    description:
        The function assume two tensor has the same rank. In case where two tensors have different
    ranks, the small one will be extend by adding redundant dimension to the beginning of axis
    (T<x>_pad_right=true) or by adding redundant dim to the last axis (if T<x>_pad_left=false
    [default]). if the Tensor #1 has shape=(i1,j1,k1,l1...), and Tensor #2 has
    shape=(i2,j2,k2,l2...); then the return Tensor will have shape=(i1*i2,j1*j2,k1*k2...)

    [Note]
        two tensor should on same device.

    */
    Tensor Kron(const Tensor &Tl, const Tensor &Tr, const bool &Tl_pad_left = false,
                const bool &Tr_pad_left = false);

    // VectorDot:
    //=================================================
    /**
    @brief perform inner product of vectors
    @param Tl Tensor #1
    @param Tr Tensor #2
    @param if the Tl should be conjugated (only work for complex. For real Tensor, no function),
    default: false
    @return
        [Tensor] Rank-0

    description:
        two Tensors must be Rank-1, with same length.

    [Note]
        performance tune: This function have better performance when two vectors with same types,
    and are one of following type: cytnx_double, cytnx_float, cytnx_complex64 or cytnx_complex128.

    */
    Tensor Vectordot(const Tensor &Tl, const Tensor &Tr, const bool &is_conj = false);

    // Dot:
    //=================================================
    /**
    @brief dot product of two arrays.
    @param Tl Tensor #1
    @param Tr Tensor #2
    @return
        [Tensor]

    description:
        1. if both Tl and Tr are 1d arrays, it is inner product of vectors (no complex conj), it
    calls linalg.Vectordot with is_conj=false.
        2. if both Tl and Tr are 2d arrays, it calls linalg.Matmul to compute the matrix
    multiplication
        3. if Tl is Nd array (with N>=2, and Tr is 1-D array, it is sum product over the last axis
    of a with b

    [Note]
        performance tune: This function have better performance when two arrays with same types, and
    are one of following type: cytnx_double, cytnx_float, cytnx_complex64 or cytnx_complex128.

    [Python]
        In Python API, operator@ is overloaded as a shorthand of linalg::Dot.
    */
    Tensor Dot(const Tensor &Tl, const Tensor &Tr);

    // Tridiag:
    //===========================================
    /**
    @brief perform diagonalization of symmetric tri-diagnoal matrix.
    @param Diag Tensor #1
    @param Sub_diag Tensor #2
    @param is_V: if calculate the eigen value.
    @param k: Return k lowest eigen vector if is_V=True
    @return
        [vector<Tensor>] if is_V = True, the first tensor is the eigen value, and second tensor is
    eigenvector of shape [k,L].

    description:
        two Tensors must be Rank-1, with length of Diag = L and Sub_diag length = L-1.

    [Note]
        performance tune: This function have better performance when two vectors with same types,
    and are one of following type: cytnx_double, cytnx_float. In general all real type can be use as
    input, which will be promote to floating point type for calculation.

    */
    std::vector<Tensor> Tridiag(const Tensor &Diag, const Tensor &Sub_diag, const bool &is_V = true,
                                const bool &is_row = false);

    // ExpH:
    //===========================================
    /**
    @brief perform matrix exponential for Hermitian matrix
    @param in input Tensor, should be Hermitian
    @param a rescale factor
    @param b bias
    @return
        [Tensor]

    description:
        perform matrix exponential with \f$O = \exp{aM + b}\f$.

    */
    Tensor ExpH(const Tensor &in, const cytnx_double &a = 1, const cytnx_double &b = 0);

    // ExpM:
    //===========================================
    /**
    @brief perform matrix exponential for generic matrix
    @param in input Tensor, should be a square rank-2.
    @param a rescale factor
    @param b bias
    @return
        [Tensor]

    description:
        perform matrix exponential with \f$O = \exp{aM + b}\f$.

    */
    Tensor ExpM(const Tensor &in, const cytnx_double &a = 1, const cytnx_double &b = 0);

    // Lanczos:
    //===========================================
    /**
    @brief perform Lanczos for hermitian/symmetric matrices or linear function.
    @param Hop the Linear Operator defined by LinOp class or it's inheritance (see LinOp).
    @param k the number of lowest k eigen values.
    @param is_V if set to true, the eigen vectors will be returned.
    @param maxiter the maximum interation steps for each k.
    @param CvgCrit the convergence criterion of the energy.
    @param is_row whether the return eigen vectors should be in row-major form.
    @param Tin the initial vector, this should be rank-1
    @param max_krydim the maximum krylov subspace dimension for each iteration.
    @param verbose print out iteration info.
    @return
        [eigvals (Tensor), eigvecs (Tensor)(option)]

    #description:
        This function calculate the eigen value problem using explicitly restarted Lanczos.

    #Performance tune:
        For small linear dimension, try to reduce max_krydim.

    #[Note]
        To use, define a linear operator with LinOp class either by assign a custom function or
    create a class that inherit LinOp (see LinOp for further details)
    */
    std::vector<Tensor> Lanczos_ER(LinOp *Hop, const cytnx_uint64 &k = 1, const bool &is_V = true,
                                   const cytnx_uint64 &maxiter = 10000,
                                   const double &CvgCrit = 1.0e-14, const bool &is_row = false,
                                   const Tensor &Tin = Tensor(), const cytnx_uint32 &max_krydim = 4,
                                   const bool &verbose = false);

    // Lanczos:
    //===========================================
    /**
    @brief perform Lanczos for hermitian/symmetric matrices or linear function to get ground state
    and lowest eigen value
    @param Hop the Linear Operator defined by LinOp class or it's inheritance (see LinOp).
    @param CvgCrit the convergence criterion of the energy.
    @param is_V if set to true, the eigen vectors will be returned.
    @param Tin the initial vector, this should be rank-1
    @param verbose print out iteration info.
    @param maxiter the maximum interation steps for each k.
    @return
        [eigvals (Tensor), eigvecs (Tensor)(option)]

    #description:
        This function calculate the eigen value problem using naive Lanczos to get ground state and
    lowest eigen value.


    #[Note]
        To use, define a linear operator with LinOp class either by assign a custom function or
    create a class that inherit LinOp (see LinOp for further details)
    */
    std::vector<Tensor> Lanczos_Gnd(LinOp *Hop, const double &CvgCrit = 1.0e-14,
                                    const bool &is_V = true, const Tensor &Tin = Tensor(),
                                    const bool &verbose = false,
                                    const unsigned int &Maxiter = 100000);

    // Lanczos:
    //===============================================
    /**
    @brief perform Lanczos for hermitian/symmetric matrices or linear function to get ground state
    and lowest eigen value
    @param Hop the Linear Operator defined by LinOp class or it's inheritance (see LinOp).
    @param CvgCrit the convergence criterion of the energy.
    @param is_V if set to true, the eigen vectors will be returned.
    @param Tin the initial vector, this should be a UniTensor.
    @param verbose print out iteration info.
    @param maxiter the maximum interation steps for each k.
    @return
        [eigvals (UniTensor::Dense), eigvecs (UniTensor)(option)]

    #description:
        This function calculate the eigen value problem using naive Lanczos to get ground state and
    lowest eigen value.


    #[Note]
        To use, define a linear operator with LinOp class either by assign a custom function or
    create a class that inherit LinOp (see LinOp for further details)
    */
    std::vector<UniTensor> Lanczos_Gnd_Ut(LinOp *Hop, const UniTensor &Tin,
                                          const double &CvgCrit = 1.0e-14, const bool &is_V = true,
                                          const bool &verbose = false,
                                          const unsigned int &Maxiter = 100000);

    // Lstsq:
    //===========================================
    /**
    @brief Return the least-squares solution to a linear matrix equation.
    @param A “Coefficient” matrix, must be two-dimensional.
    @param b Ordinate or “dependent variable” values, must be two-dimensional, the least-squares
    solution is calculated for each of the K columns of b.
    @param rcond Cut-off ratio for small singular values of a. For the purposes of rank
    determination, singular values are treated as zero if they are smaller than rcond times the
    largest singular value of A, If it is negative, the machine precision is used.
    @return [std::vector<Tensors>]

        1. the first tensor is least-squares solutions in the K columns.
        2. the second tensor is the sums of squared residuals: Squared Euclidean 2-norm for each
    column in b - a @ x. If the rank of a is < N or M <= N, this is a zero Tensor.
        3. the third tensor is the rank of matrix A.
        4. the forth tensor is singular values of A.

    #description:
        Computes the vector x that approximatively solves the equation A @ x = b. The equation may
    be under-, well-, or over-determined independent columns. If a is square and of full rank, then
    x (but for round-off error) is the “exact” solution of the equation. Else, x minimizes the
    Euclidean 2-norm || b - a x ||.

    [Ke]
    */
    std::vector<Tensor> Lstsq(const Tensor &A, const Tensor &b, const float &rcond = -1);

  }  // namespace linalg

  // operators:
  Tensor operator+(const Tensor &Lt, const Tensor &Rt);
  template <class T>
  Tensor operator+(const T &lc, const Tensor &Rt);
  template <class T>
  Tensor operator+(const Tensor &Lt, const T &rc);

  //------------------------------------
  Tensor operator-(const Tensor &Lt, const Tensor &Rt);
  template <class T>
  Tensor operator-(const T &lc, const Tensor &Rt);
  template <class T>
  Tensor operator-(const Tensor &Lt, const T &rc);

  //-----------------------------------
  Tensor operator*(const Tensor &Lt, const Tensor &Rt);
  template <class T>
  Tensor operator*(const T &lc, const Tensor &Rt);
  template <class T>
  Tensor operator*(const Tensor &Lt, const T &rc);

  //----------------------------------
  Tensor operator/(const Tensor &Lt, const Tensor &Rt);
  template <class T>
  Tensor operator/(const T &lc, const Tensor &Rt);
  template <class T>
  Tensor operator/(const Tensor &Lt, const T &rc);

  //----------------------------------
  Tensor operator%(const Tensor &Lt, const Tensor &Rt);
  template <class T>
  Tensor operator%(const T &lc, const Tensor &Rt);
  template <class T>
  Tensor operator%(const Tensor &Lt, const T &rc);

  //----------------------------------
  Tensor operator==(const Tensor &Lt, const Tensor &Rt);
  template <class T>
  Tensor operator==(const T &lc, const Tensor &Rt);
  template <class T>
  Tensor operator==(const Tensor &Lt, const T &rc);

}  // namespace cytnx

#endif
