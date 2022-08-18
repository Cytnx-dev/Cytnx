#ifndef _linalg_H_
#define _linalg_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include "Storage.hpp"
#include "linalg/linalg_internal_interface.hpp"

namespace cytnx {
  /**
  @namespace cytnx::linalg
  @brief linear algebra related functions.
  */
  namespace linalg {
    // Add:
    //==================================================
    /**
    @brief element-wise add
    */
    Tensor Add(const Tensor &Lt, const Tensor &Rt);
    template <class T>
    Tensor Add(const T &lc, const Tensor &Rt);
    template <class T>
    Tensor Add(const Tensor &Lt, const T &rc);

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
                                     const bool &is_U = true, const bool &is_vT = true);

    // Eig:
    //==================================================
    /**
    @brief eigen-value decomposition for Hermitian matrix

    [Note] the Tin should be a rank-2 Tensor.
    */
    std::vector<Tensor> Eigh(const Tensor &Tin, const bool &is_V = false);

    // Matmul:
    //==================================================
    /**
    @brief perform matrix multiplication on two tensors.

    [Note] the TL and TR should be both rank-2 Tensor.
    */
    Tensor Matmul(const Tensor &TL, const Tensor &TR);

    // Inv:
    //==================================================
    /**
    @brief Matrix inverse.
    @return
        [Tensor]

    [Note] the Tin should be a rank-2 Tensor.
    */
    Tensor Inv(const Tensor &Tin);
    /**
    @brief inplace perform Matrix inverse.

    description:
        on return, the Tin will be modified to it's inverse.

    [Note] the Tin should be a rank-2 Tensor.
    */
    void Inv_(Tensor &Tin);

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
        [Tensor]

    */
    Tensor Exp(const Tensor &Tin);
    /**
    @brief inplace perform Exponential on all the element in Tensor.
    @param Tin, the input Tensor.

    description:
        on return, the elements in Tin will be modified to it's exponetial value.
    */
    void Exp_(Tensor &Tin);

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
    @return
        [Tensor]

    [Note]
        1. the elements in idxl and idxr have one to one correspondence.
        2. two tensors should on same device.
    */
    Tensor Tensordot(const Tensor &Tl, const Tensor &Tr, const std::vector<cytnx_uint64> &idxl,
                     const std::vector<cytnx_uint64> &idxr);

    // Otimes:
    //==================================================
    /**
    @brief perform outer produces of two Tensor.
    @param Tl Tensor #1
    @param Tr Tensor #2
    @return
        [Tensor]

    description:
        if the Tensor #1 has [shape_1], and Tensor #2 has [shape_2]; then the return Tensor will
    have shape: concate(shape_1,shape_2)

    [Note]
        two tensor should on same device.

    */
    Tensor Otimes(const Tensor &Tl, const Tensor &Tr);

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
  Tensor operator==(const Tensor &Lt, const Tensor &Rt);
  template <class T>
  Tensor operator==(const T &lc, const Tensor &Rt);
  template <class T>
  Tensor operator==(const Tensor &Lt, const T &rc);

}  // namespace cytnx

#endif
