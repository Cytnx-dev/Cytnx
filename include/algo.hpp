#ifndef _algo_H_
#define _algo_H_
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  /**
   * @namespace cytnx::algo
   * @brief Some basic algorithms API.
   */
  namespace algo {

    /**
    @brief sort Tensor along last axis.
    @param[in] Tin the input Tensor.
    @return [Tensor]
    */
    Tensor Sort(const Tensor &Tin);

    /**
     * @brief concatenate two 1d Tensor.
     * @param[in] T1 the first Tensor.
     * @param[in] T2 the second Tensor.
     * @return [Tensor] return a new Tensor that is the concatenation of T1 and T2.
     * @pre
     * 1. \p T1 and \p T2 should be 1d Tensor.
     * 2. \p T1 and \p T2 should be on same device.
     * 3. The data type of \p T1 and \p T2 cannot be Type.Void.
     * @note
     * If the data type of \p T1 and \p T2 are different, the data type of the
     *   output Tensor will be the stronger one.
     */
    Tensor Concatenate(Tensor T1, Tensor T2);

    /**
     * @brief vertical stack a list of Tensor.
     * @details This function will stack vertically a list of Matrices (a 2d tensor)
     *   with same number of columns. That is, if the input Matrices are
     * \f$ [M_1, M_2, M_3] \f$, the output will be:
     * \f[
     * \begin{bmatrix}
     * M_1\\
     * M_2\\
     * M_3\\
     * \end{bmatrix}
     * \f]
     * Note that the number of columns of \f$ M_1 \f$, \f$ M_2 \f$ and \f$ M_3 \f$
     *  should be the same.
     * @param[in] In_tensors a list of 2d Tensor (Matrix).
     * @return [Tensor] return a new Tensor that is the vertical stack of
     *  \p In_tensors.
     * @pre
     * 1. The input \p In_tensors cannot be empty.
     * 2. The input \p In_tensors need to be 2d Tensor (Matrix).
     * 3. The input \p In_tensors should have same number of columns.
     * 4. The input \p In_tensors should be on same device.
     * 5. The data type of the input \p In_tensors cannot be Type.Bool or Type.Void.
     *   (see cytnx::Type)
     * @note
     * If the data type of the input \p In_tensors are different, the data type of the
     *   output Tensor will be the stronger one.
     */
    Tensor Vstack(const std::vector<Tensor> &In_tensors);

    /**
     * @brief horizontal stack a list of Tensor.
     * @details This function will stack horizontally a list of Matrices (a 2d tensor)
     *   with same number of rows. That is, if the input Matrices are
     * \f$ [M_1, M_2, M_3] \f$, the output will be:
     * \f[
     * \begin{bmatrix}
     * M_1, M_2, M_3
     * \end{bmatrix}
     * \f]
     * Note that the number of rows of \f$ M_1 \f$, \f$ M_2 \f$ and \f$ M_3 \f$
     *  should be the same.
     * @param[in] In_tensors a list of 2d Tensor (Matrix).
     * @return [Tensor] return a new Tensor that is the horizontal stack of
     *  \p In_tensors.
     * @pre
     * 1. The input \p In_tensors cannot be empty.
     * 2. The input \p In_tensors need to be 2d Tensor (Matrix).
     * 3. The input \p In_tensors should have same number of rows.
     * 4. The input \p In_tensors should be on same device.
     * 5. The data type of the input \p In_tensors cannot be Type.Bool or Type.Void.
     *   (see cytnx::Type)
     * @note
     * If the data type of the input \p In_tensors are different, the data type of the
     *   output Tensor will be the stronger one.
     */
    Tensor Hstack(const std::vector<Tensor> &In_tensors);

    /**
     * @brief split a Matrix (a 2d Tensor) into a list of Matrices along the vertical direction.
     * @details This function is same as
     *   @ref Vsplit(const Tensor &, const std::vector<cytnx_uint64> &) "Vsplit",
     *   but the output is in the argument.
     * @param[out] out a list of Tensor that will be the output.
     * @param[in] Tin the input Tensor.
     * @param[in] dims the row number of each Matrix in the output list.
     * @see Vsplit
     */
    void Vsplit_(std::vector<Tensor> &out, const Tensor &Tin,
                 const std::vector<cytnx_uint64> &dims);

    /**
     * @brief split a Matrix (a 2d Tensor) into a list of Matrices along the horizontal direction.
     * @details This function is same as
     *   @ref Hsplit(const Tensor &, const std::vector<cytnx_uint64> &) "Hsplit",
     *   but the output is in the argument.
     * @param[out] out a list of Tensor that will be the output.
     * @param[in] Tin the input Tensor.
     * @param[in] dims the column number of each Matrix in the output list.
     * @see Hsplit
     */
    void Hsplit_(std::vector<Tensor> &out, const Tensor &Tin,
                 const std::vector<cytnx_uint64> &dims);

    /**
     * @brief split a Matrix (a 2d Tensor) into a list of Matrices along the vertical direction.
     * @details This function will split a Matrix (a 2d Tensor) into a list of Matrices
     *  along the vertical direction. That is, if the input Matrix \p Tin is:
     * \f[
     * M = \begin{bmatrix}
     * m_{11} & m_{12} & m_{13}\\
     * m_{21} & m_{22} & m_{23}\\
     * m_{31} & m_{32} & m_{33}\\
     * \end{bmatrix}
     * \f]
     * and the input \p dims is \f$ [1,2] \f$, the output will be \f$ [M_1, M_2] \f$,
     * where:
     * \f[
     * M_1 = \begin{bmatrix}
     * m_{11} & m_{12} & m_{13}\\
     * \end{bmatrix}
     * \f]
     * and
     * \f[
     * M_2 = \begin{bmatrix}
     * m_{21} & m_{22} & m_{23}\\
     * m_{31} & m_{32} & m_{33}\\
     * \end{bmatrix}
     * \f]
     *
     * @param[in] Tin the input Matrix (a 2d Tensor).
     * @param[in] dims the row number of each Matrix in the output list.
     * @return [std::vector<Tensor>] return a list of Matrices that is the split of
     * \p Tin with respect to \p dims.
     * @pre
     * 1. The input \p Tin need to be 2d Tensor (Matrix).
     * 2. The input \p dims cannot be empty.
     * 3. The input \p dims cannot be out of range. That is, the summation of \p dims
     *  need equal to the number of rows of \p Tin.
     * 4. The elements in \p dims cannot be zero.
     * @see Hsplit, Vstack
     */
    std::vector<Tensor> Vsplit(const Tensor &Tin, const std::vector<cytnx_uint64> &dims);

    /**
     * @brief split a Matrix (a 2d Tensor) into a list of Matrices along the horizontal direction.
     * @details This function will split a Matrix (a 2d Tensor) into a list of Matrices
     *  along the horizontal direction. That is, if the input Matrix \p Tin is:
     * \f[
     * M = \begin{bmatrix}
     * m_{11} & m_{12} & m_{13}\\
     * m_{21} & m_{22} & m_{23}\\
     * m_{31} & m_{32} & m_{33}\\
     * \end{bmatrix}
     * \f]
     * and the input \p dims is \f$ [2,1] \f$, the output will be \f$ [M_1, M_2] \f$,
     * where:
     * \f[
     * M_1 = \begin{bmatrix}
     * m_{11} & m_{12}\\
     * m_{21} & m_{22}\\
     * m_{31} & m_{32}\\
     * \end{bmatrix}
     * \f]
     * and
     * \f[
     * M_2 = \begin{bmatrix}
     * m_{13}\\
     * m_{23}\\
     * m_{33}\\
     * \end{bmatrix}
     * \f]
     *
     * @param[in] Tin the input Matrix (a 2d Tensor).
     * @param[in] dims the column number of each Matrix in the output list.
     * @return [std::vector<Tensor>] return a list of Matrices that is the split of
     * \p Tin with respect to \p dims.
     * @pre
     * 1. The input \p Tin need to be 2d Tensor (Matrix).
     * 2. The input \p dims cannot be empty.
     * 3. The input \p dims cannot be out of range. That is, the summation of \p dims
     *  need equal to the number of columns of \p Tin.
     * 4. The elements in \p dims cannot be zero.
     * @see Vsplit, Hstack
     */
    std::vector<Tensor> Hsplit(const Tensor &Tin, const std::vector<cytnx_uint64> &dims);

    ///@cond
    Tensor _fx_Matric_combine(std::vector<Tensor> &Tlist, const cytnx_uint64 &Row_Nblk,
                              const cytnx_uint64 &Col_Nblk);
    void _fx_Matric_split(std::vector<Tensor> &out, const Tensor &Tin,
                          const std::vector<cytnx_uint64> &rdims,
                          const std::vector<cytnx_uint64> &cdims);
    ///@endcond

  }  // namespace algo
}  // namespace cytnx
#endif  // BACKEND_TORCH

#endif
