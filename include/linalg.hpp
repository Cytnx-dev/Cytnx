#ifndef _linalg_H_
#define _linalg_H_

#include "LinOp.hpp"
#include "Tensor.hpp"
#include "Type.hpp"
#include "UniTensor.hpp"
#include "cytnx_error.hpp"

#ifdef BACKEND_TORCH
#else

  #include <functional>

  #include "backend/Scalar.hpp"
  #include "backend/Storage.hpp"

namespace cytnx {
  int set_mkl_ilp64();
  int get_mkl_code();

  /**
   * @brief The addtion operator between two UniTensor.
   * @details This is the addtion function for UniTensor. It will call
   * linalg::Add(const UniTensor &Lt, const UniTensor &Rt) function.
   * @param[in] Lt The left UniTensor.
   * @param[in] Rt The right UniTensor.
   * @return [UniTensor] The result of the addtion.
   * @pre \p Lt and \p Rt must have the same shape.
   * @see linalg::Add(const UniTensor &Lt, const UniTensor &Rt)
   */
  cytnx::UniTensor operator+(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);

  /**
   * @brief The addtion operator between a template type and a UniTensor.
   * @details This is the addtion function for UniTensor. It will call
   * linalg::Add(const T &lc, const UniTensor &Rt) function.
   * @param[in] lc The left template type.
   * @param[in] Rt The right UniTensor.
   * @return [UniTensor] The result of the addtion.
   * @see linalg::Add(const T &lc, const UniTensor &Rt)
   */
  template <class T>
  cytnx::UniTensor operator+(const T &lc, const cytnx::UniTensor &Rt);

  /**
   * @brief The addtion operator between a UniTensor and a template type.
   * @details This is the addtion function for UniTensor. It will call
   * linalg::Add(const UniTensor &Lt, const T &rc) function.
   * @param[in] Lt The left UniTensor.
   * @param[in] rc The right template type.
   * @return [UniTensor] The result of the addtion.
   * @see linalg::Add(const UniTensor &Lt, const T &rc)
   */
  template <class T>
  cytnx::UniTensor operator+(const cytnx::UniTensor &Lt, const T &rc);

  /**
   * @brief The subtraction operator between two UniTensor.
   * @details This is the subtraction function for UniTensor. It will call
   * linalg::Sub(const UniTensor &Lt, const UniTensor &Rt) function.
   * @param[in] Lt The left UniTensor.
   * @param[in] Rt The right UniTensor.
   * @return [UniTensor] The result of the subtraction.
   * @pre \p Lt and \p Rt must have the same shape.
   * @see linalg::Sub(const UniTensor &Lt, const UniTensor &Rt)
   */
  cytnx::UniTensor operator-(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);

  /**
   * @brief The subtraction operator between a template type and a UniTensor.
   * @details This is the subtraction function for UniTensor. It will call
   * linalg::Sub(const T &lc, const UniTensor &Rt) function.
   * @param[in] lc The left template type.
   * @param[in] Rt The right UniTensor.
   * @return [UniTensor] The result of the subtraction.
   * @see linalg::Sub(const T &lc, const UniTensor &Rt)
   */
  template <class T>
  cytnx::UniTensor operator-(const T &lc, const cytnx::UniTensor &Rt);

  /**
   * @brief The subtraction operator between a UniTensor and a template type.
   * @details This is the subtraction function for UniTensor. It will call
   * linalg::Sub(const UniTensor &Lt, const T &rc) function.
   * @param[in] Lt The left UniTensor.
   * @param[in] rc The right template type.
   * @return [UniTensor] The result of the subtraction.
   * @see linalg::Sub(const UniTensor &Lt, const T &rc)
   */
  template <class T>
  cytnx::UniTensor operator-(const cytnx::UniTensor &Lt, const T &rc);

  /**
   * @brief The multiplication operator between two UniTensor.
   * @details This is the multiplication function for UniTensor. It will call
   * linalg::Mul(const UniTensor &Lt, const UniTensor &Rt) function.
   * @param[in] Lt The left UniTensor.
   * @param[in] Rt The right UniTensor.
   * @return [UniTensor] The result of the multiplication.
   * @pre \p Lt and \p Rt must have the same shape.
   * @see linalg::Mul(const UniTensor &Lt, const UniTensor &Rt)
   */
  cytnx::UniTensor operator*(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);

  /**
   * @brief The multiplication operator between a template type and a UniTensor.
   * @details This is the multiplication function for UniTensor. It will call
   * linalg::Mul(const T &lc, const UniTensor &Rt) function.
   * @param[in] lc The left template type.
   * @param[in] Rt The right UniTensor.
   * @return [UniTensor] The result of the multiplication.
   * @see linalg::Mul(const T &lc, const UniTensor &Rt)
   */
  template <class T>
  cytnx::UniTensor operator*(const T &lc, const cytnx::UniTensor &Rt);

  /**
   * @brief The multiplication operator between a UniTensor and a template type.
   * @details This is the multiplication function for UniTensor. It will call
   * linalg::Mul(const UniTensor &Lt, const T &rc) function.
   * @param[in] Lt The left UniTensor.
   * @param[in] rc The right template type.
   * @return [UniTensor] The result of the multiplication.
   * @see linalg::Mul(const UniTensor &Lt, const T &rc)
   */
  template <class T>
  cytnx::UniTensor operator*(const cytnx::UniTensor &Lt, const T &rc);

  /**
   * @brief The division operator between two UniTensor.
   * @details This is the division function for UniTensor. It will call
   * linalg::Div(const UniTensor &Lt, const UniTensor &Rt) function.
   * @param[in] Lt The left UniTensor.
   * @param[in] Rt The right UniTensor.
   * @return [UniTensor] The result of the division.
   * @pre \p Lt and \p Rt must have the same shape.
   * @see linalg::Div(const UniTensor &Lt, const UniTensor &Rt)
   */
  cytnx::UniTensor operator/(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);

  /**
   * @brief The division operator between a template type and a UniTensor.
   * @details This is the division function for UniTensor. It will call
   * linalg::Div(const T &lc, const UniTensor &Rt) function.
   * @param[in] lc The left template type.
   * @param[in] Rt The right UniTensor.
   * @return [UniTensor] The result of the division.
   * @see linalg::Div(const T &lc, const UniTensor &Rt)
   */
  template <class T>
  cytnx::UniTensor operator/(const T &lc, const cytnx::UniTensor &Rt);

  /**
   * @brief The division operator between a UniTensor and a template type.
   * @details This is the division function for UniTensor. It will call
   * linalg::Div(const UniTensor &Lt, const T &rc) function.
   * @param[in] Lt The left UniTensor.
   * @param[in] rc The right template type.
   * @return [UniTensor] The result of the division.
   * @see linalg::Div(const UniTensor &Lt, const T &rc)
   */
  template <class T>
  cytnx::UniTensor operator/(const cytnx::UniTensor &Lt, const T &rc);

  /**
   * @brief The modulo operator between two UniTensor.
   * @details This is the modulo function for UniTensor. It will call
   * linalg::Mod(const UniTensor &Lt, const UniTensor &Rt) function.
   * @param[in] Lt The left UniTensor.
   * @param[in] Rt The right UniTensor.
   * @return [UniTensor] The result of the modulo.
   * @pre \p Lt and \p Rt must have the same shape.
   * @see linalg::Mod(const UniTensor &Lt, const UniTensor &Rt)
   */
  cytnx::UniTensor operator%(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);

  /**
   * @brief The modulo operator between a template type and a UniTensor.
   * @details This is the modulo function for UniTensor. It will call
   * linalg::Mod(const T &lc, const UniTensor &Rt) function.
   * @param[in] lc The left template type.
   * @param[in] Rt The right UniTensor.
   * @return [UniTensor] The result of the modulo.
   * @see linalg::Mod(const T &lc, const UniTensor &Rt)
   */
  template <class T>
  cytnx::UniTensor operator%(const T &lc, const cytnx::UniTensor &Rt);

  /**
   * @brief The modulo operator between a UniTensor and a template type.
   * @details This is the modulo function for UniTensor. It will call
   * linalg::Mod(const UniTensor &Lt, const T &rc) function.
   * @param[in] Lt The left UniTensor.
   * @param[in] rc The right template type.
   * @return [UniTensor] The result of the modulo.
   * @see linalg::Mod(const UniTensor &Lt, const T &rc)
   */
  template <class T>
  cytnx::UniTensor operator%(const cytnx::UniTensor &Lt, const T &rc);

  /**
  @namespace cytnx::linalg
  @brief linear algebra related functions.
  @details This namespace contains all the linear algebra related functions.
  For example, the matrix multiplication, the singular value decomposition, etc.
  If the linear algebra can only perfom on a matrix, then in most cases,
  1. If the object is Tensor, then it need to be rank-2.
  2. If the object is UniTensor, then the result will depend on the UniTensor's
  rowrank.
  */
  namespace linalg {

    // Add:
    //==================================================
    /**
     * @brief The addtion function between two UniTensor.
     * @details This is the addtion function for UniTensor. It will perform
     * the element-wise addtion. That means if the left UniTensor \p Lt
     * is given as \f$ T_L \f$ and the right UniTensor \p Rt is given as \f$ T_R \f$,
     * then the result will be:
     *   \f[
     *    T_o[i] = T_L[i] + T_R[i],
     *  \f]
     * where \f$ T_L[i] \f$ and \f$ T_R[i] \f$ are the elements in the
     * UniTensor \f$ T_L \f$ and \f$ T_R \f$.
     * It will perform the element-wise addtion and note that it will return a
     * new UniTensor object.
     * @param[in] Lt The left UniTensor.
     * @param[in] Rt The right UniTensor.
     * @return The result UniTensor.
     * @pre \p Lt and \p Rt must have the same shape.
     * @see
     *     UniTensor::Add(const UniTensor &Rt) const,
     *     operator+(const UniTensor &Lt, const UniTensor &Rt)
     */
    cytnx::UniTensor Add(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);

    /**
     * @brief The addtion function between a template type and a UniTensor.
     * @details This is the addtion function for UniTensor. It will
     * add the UniTensor and a template type together and add every element
     * in the UniTensor with the template type. That means if the template type
     * \p lc is given as \f$ c \f$ and the UniTensor \p Rt is given as \f$ T_i \f$,
     * then the result will be:
     * \f[
     *    T_o[i] = c + T_i[i],
     * \f]
     * where \f$ T_i[i] \f$ and \f$ T_o[i] \f$ are the elements in the
     * UniTensor \f$ T_i \f$ and \f$ T_o \f$.
     * @param[in] lc
     * @parblock
     * The left template type.
     *
     * supported type:
     * Scalar, cytnx::cytnx_complex128, cytnx::cytnx_complex64,
     * cytnx::cytnx_double, cytnx::cytnx_float, cytnx::cytnx_int64,
     * cytnx::cytnx_uint64, cytnx::cytnx_int32, cytnx::cytnx_uint32,
     * cytnx::cytnx_int16, cytnx::cytnx_uint16, cytnx::cytnx_bool.
     * @endparblock
     * @param[in] Rt The right UniTensor.
     * @return The result UniTensor.
     * @pre The supported template type shown above.
     * @note
     * The inpute template type \p lc will be casted to the same type as
     * the UniTensor \p Rt.
     * @see
     *    operator+(const T &lc, const UniTensor &Rt),
     *    Add(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt)
     */
    template <class T>
    cytnx::UniTensor Add(const T &lc, const cytnx::UniTensor &Rt);

    /**
     * @brief The addtion function between a UniTensor and a template type.
     * @details This is the addtion function for UniTensor. It will
     * add the UniTensor and a template type together and add every element
     * in the UniTensor with the template type. That means if the UniTensor
     * \p Lt is given as \f$ T_i \f$ and the template type \p rc is given as
     *  \f$ c \f$, then the result will be:
     * \f[
     *   T_o[i] = T_i[i] + c,
     * \f]
     * where \f$ T_i[i] \f$ and \f$ T_o[i] \f$ are the elements in the
     * UniTensor \f$ T_i \f$ and \f$ T_o \f$.
     *
     * @param[in] Lt The left UniTensor.
     * @param[in] rc
     * @parblock
     * The right template type.
     *
     * supported type:
     * Scalar, cytnx::cytnx_complex128, cytnx::cytnx_complex64,
     * cytnx::cytnx_double, cytnx::cytnx_float, cytnx::cytnx_int64,
     * cytnx::cytnx_uint64, cytnx::cytnx_int32, cytnx::cytnx_uint32,
     * cytnx::cytnx_int16, cytnx::cytnx_uint16, cytnx::cytnx_bool.
     * @endparblock
     * @return The result UniTensor.
     * @pre The supported template type shown above.
     * @note
     * The inpute template type \p rc will be casted to the same type as
     * the UniTensor \p Lt.
     * @see
     * operator+(const UniTensor &Lt, const T &rc),
     * Add(const T &lc, const cytnx::UniTensor &Rt)
     */
    template <class T>
    cytnx::UniTensor Add(const cytnx::UniTensor &Lt, const T &rc);

    // Sub:
    //==================================================
    /**
     * @brief The subtraction function between two UniTensor.
     * @details This is the subtraction function for UniTensor. It will
     * subtract the UniTensor and a template type together. It will subtract every element
     * in the UniTensor with the template type. That means if the UniTensor
     * \p Lt is given as \f$ T_L \f$ and the UniTensor \p Rt is given as \f$ T_R \f$,
     * then the result will be:
     * \f[
     *  T_o[i] = T_L[i] - T_R[i],
     * \f]
     * where \f$ T_L[i] \f$ and \f$ T_R[i] \f$ are the elements in the
     * UniTensor \f$ T_L \f$ and \f$ T_R \f$.
     * It will perform the element-wise subtraction and note that it will return a
     * new UniTensor object.
     * @param[in] Lt The left UniTensor.
     * @param[in] Rt The right UniTensor.
     * @return The result UniTensor.
     * @pre \p Lt and \p Rt must have the same shape.
     * @see
     *     UniTensor::Sub(const UniTensor &Rt) const,
     *     operator-(const UniTensor &Lt, const UniTensor &Rt)
     */
    cytnx::UniTensor Sub(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);

    /**
     * @brief The subtraction function between a UniTensor and a template type.
     * @details This is the subtraction function for UniTensor. It will
     * subtract the UniTensor and a template type together. It will subtract every element
     * in the UniTensor with the template type. That means if the template type
     * \p lc is given as \f$ c \f$ and the UniTensor \p Rt is given as \f$ T_i \f$,
     * then the result will be:
     * \f[
     *  T_o[i] = c - T_i[i],
     * \f]
     * where \f$ T_i[i] \f$ and \f$ T_o[i] \f$ are the elements in the
     * UniTensor \f$ T_i \f$ and \f$ T_o \f$.
     * @param[in] Lt The left UniTensor.
     * @param[in] rc
     * @parblock
     * The right template type.
     *
     * supported type:
     * Scalar, cytnx::cytnx_complex128, cytnx::cytnx_complex64,
     * cytnx::cytnx_double, cytnx::cytnx_float, cytnx::cytnx_int64,
     * cytnx::cytnx_uint64, cytnx::cytnx_int32, cytnx::cytnx_uint32,
     * cytnx::cytnx_int16, cytnx::cytnx_uint16, cytnx::cytnx_bool.
     * @endparblock
     * @return The result UniTensor.
     * @pre The supported template type shown above.
     * @note
     * The inpute template type \p lc will be casted to the same type as
     * the UniTensor \p Rt.
     * @see
     *    operator-(const T &lc, const UniTensor &Rt),
     *    Sub(const T &lc, const cytnx::UniTensor &Rt)
     */
    template <class T>
    cytnx::UniTensor Sub(const T &lc, const cytnx::UniTensor &Rt);

    /**
     * @brief The subtraction function between a UniTensor and a template type.
     * @details This is the subtraction function for UniTensor. It will
     * subtract the UniTensor and a template type together. It will subtract every element
     * in the UniTensor with the template type. That means if the UniTensor
     * \p Lt is given as \f$ T_i \f$ and the template type \p rc is given as
     *  \f$ c \f$, then the result will be:
     * \f[
     *   T_o[i] = T_i[i] - c,
     * \f]
     * where \f$ T_i[i] \f$ and \f$ T_o[i] \f$ are the elements in the
     * UniTensor \f$ T_i \f$ and \f$ T_o \f$.
     * @param[in] Lt The left UniTensor.
     * @param[in] rc
     * @parblock
     * The right template type.
     *
     * supported type:
     * Scalar, cytnx::cytnx_complex128, cytnx::cytnx_complex64,
     * cytnx::cytnx_double, cytnx::cytnx_float, cytnx::cytnx_int64,
     * cytnx::cytnx_uint64, cytnx::cytnx_int32, cytnx::cytnx_uint32,
     * cytnx::cytnx_int16, cytnx::cytnx_uint16, cytnx::cytnx_bool.
     * @endparblock
     * @return The result UniTensor.
     * @pre The supported template type shown above.
     * @note
     * The inpute template type \p rc will be casted to the same type as
     * the UniTensor \p Lt.
     * @see
     *    operator-(const UniTensor &Lt, const T &rc),
     *    Sub(const cytnx::UniTensor &Lt, const T &rc)
     */
    template <class T>
    cytnx::UniTensor Sub(const cytnx::UniTensor &Lt, const T &rc);

    // Mul:
    //==================================================
    /**
     * @brief The multiplication function between two UniTensor.
     * @details This is the multiplication function for UniTensor. It will
     * multiply the two UniTensor together. It will multiply every element
     * in the UniTensor \p Lt with the corresponding element in the UniTensor
     * \p Rt. That means if the UniTensor \p Lt is given as \f$ T_L \f$ and the UniTensor
     * \p Rt is given as \f$ T_R \f$, then the result will be:
     * \f[
     *  T_o[i] = T_L[i] \times T_R[i],
     * \f]
     * where \f$ T_L[i] \f$ and \f$ T_R[i] \f$ are the elements in the
     * UniTensor \f$ T_L \f$ and \f$ T_R \f$.
     * It will perform the element-wise multiplication and note that it will return a
     * new UniTensor object.
     * @param[in] Lt The left UniTensor.
     * @param[in] Rt The right UniTensor.
     * @return The result UniTensor.
     * @pre \p Lt and \p Rt must have the same shape.
     * @see
     *     UniTensor::Mul(const UniTensor &Rt) const,
     *     operator*(const UniTensor &Lt, const UniTensor &Rt)
     */
    cytnx::UniTensor Mul(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);

    /**
     * @brief The multiplication function between a UniTensor and a template type.
     * @details This is the multiplication function for UniTensor. It will
     * multiply the UniTensor and a template type together. It will multiply every element
     * in the UniTensor with the template type. That means if the template type
     * \p lc is given as \f$ c \f$ and the UniTensor \p Rt is given as \f$ T_i \f$,
     * then the result will be:
     * \f[
     *  T_o[i] = c \times T_i[i],
     * \f]
     * where \f$ T_i[i] \f$ and \f$ T_o[i] \f$ are the elements in the
     * UniTensor \f$ T_i \f$ and \f$ T_o \f$.
     * @param[in] Lt The left UniTensor.
     * @param[in] rc
     * @parblock
     * The right template type.
     *
     * supported type:
     * Scalar, cytnx::cytnx_complex128, cytnx::cytnx_complex64,
     * cytnx::cytnx_double, cytnx::cytnx_float, cytnx::cytnx_int64,
     * cytnx::cytnx_uint64, cytnx::cytnx_int32, cytnx::cytnx_uint32,
     * cytnx::cytnx_int16, cytnx::cytnx_uint16, cytnx::cytnx_bool.
     * @endparblock
     * @return The result UniTensor.
     * @pre The supported template type shown above.
     * @note
     * The inpute template type \p lc will be casted to the same type as
     * the UniTensor \p Rt.
     * @see
     *    operator*(const T &lc, const UniTensor &Rt),
     *    Mul(const T &lc, const cytnx::UniTensor &Rt)
     */
    template <class T>
    cytnx::UniTensor Mul(const T &lc, const cytnx::UniTensor &Rt);

    /**
     * @brief The multiplication function between a UniTensor and a template type.
     * @details This is the multiplication function for UniTensor. It will
     * multiply the UniTensor and a template type together. It will multiply every element
     * in the UniTensor with the template type. That means if the UniTensor
     * \p Lt is given as \f$ T_i \f$ and the template type \p rc is given as
     *  \f$ c \f$, then the result will be:
     * \f[
     *   T_o[i] = T_i[i] \times c,
     * \f]
     * where \f$ T_i[i] \f$ and \f$ T_o[i] \f$ are the elements in the
     * UniTensor \f$ T_i \f$ and \f$ T_o \f$.
     * @param[in] Lt The left UniTensor.
     * @param[in] rc
     * @parblock
     * The right template type.
     *
     * supported type:
     * Scalar, cytnx::cytnx_complex128, cytnx::cytnx_complex64,
     * cytnx::cytnx_double, cytnx::cytnx_float, cytnx::cytnx_int64,
     * cytnx::cytnx_uint64, cytnx::cytnx_int32, cytnx::cytnx_uint32,
     * cytnx::cytnx_int16, cytnx::cytnx_uint16, cytnx::cytnx_bool.
     * @endparblock
     * @return The result UniTensor.
     * @pre The supported template type shown above.
     * @note
     * The inpute template type \p rc will be casted to the same type as
     * the UniTensor \p Lt.
     * @see
     *    operator*(const UniTensor &Lt, const T &rc),
     *    Mul(const cytnx::UniTensor &Lt, const T &rc)
     */
    template <class T>
    cytnx::UniTensor Mul(const cytnx::UniTensor &Lt, const T &rc);

    // Div:
    //==================================================
    /**
     * @brief The division function between two UniTensor.
     * @details This is the division function for UniTensor. It will
     * divide the left UniTensor and the right UniTensor together. It will divide every element
     * in the left UniTensor with the right UniTensor. That means if the left UniTensor
     * \p Lt is given as \f$ T_L \f$ and the right UniTensor \p Rt is given as
     *  \f$ T_R \f$, then the result will be:
     * \f[
     *   T_o[i] = T_L[i] \div T_R[i],
     * \f]
     * where \f$ T_L[i] \f$ and \f$ T_R[i] \f$ are the elements in the
     * UniTensor \f$ T_L \f$ and \f$ T_R \f$.
     * It will perform the element-wise division and note that it will return a
     * new UniTensor object.
     * @param[in] Lt The left UniTensor.
     * @param[in] Rt The right UniTensor.
     * @return The result UniTensor.
     * @pre \p Lt and \p Rt must have the same shape.
     * @see
     *     UniTensor::Div(const UniTensor &Rt) const,
     *     operator/(const UniTensor &Lt, const UniTensor &Rt)
     */
    cytnx::UniTensor Div(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);

    /**
     * @brief The division function between a UniTensor and a template type.
     * @details This is the division function for UniTensor. It will
     * divide the UniTensor and a template type together. It will divide every element
     * in the UniTensor with the template type. That means if the template type
     * \p lc is given as \f$ c \f$ and the UniTensor \p Rt is given as \f$ T_i \f$,
     * then the result will be:
     * \f[
     *  T_o[i] = c \div T_i[i],
     * \f]
     * where \f$ T_i[i] \f$ and \f$ T_o[i] \f$ are the elements in the
     * UniTensor \f$ T_i \f$ and \f$ T_o \f$.
     * @param[in] Lt The left UniTensor.
     * @param[in] rc
     * @parblock
     * The right template type.
     *
     * supported type:
     * Scalar, cytnx::cytnx_complex128, cytnx::cytnx_complex64,
     * cytnx::cytnx_double, cytnx::cytnx_float, cytnx::cytnx_int64,
     * cytnx::cytnx_uint64, cytnx::cytnx_int32, cytnx::cytnx_uint32,
     * cytnx::cytnx_int16, cytnx::cytnx_uint16, cytnx::cytnx_bool.
     * @endparblock
     * @return The result UniTensor.
     * @pre The supported template type shown above.
     * @note
     * 1. The inpute template type \p lc will be casted to the same type as
     * the UniTensor \p Rt.
     * 2. The division by zero is not allowed.
     * @see
     *    operator/(const T &lc, const UniTensor &Rt),
     *    Div(const T &lc, const cytnx::UniTensor &Rt)
     */
    template <class T>
    cytnx::UniTensor Div(const T &lc, const cytnx::UniTensor &Rt);

    /**
     * @brief The division function between a UniTensor and a template type.
     * @details This is the division function for UniTensor. It will
     * divide the UniTensor and a template type together. It will divide every element
     * in the UniTensor with the template type. That means if the UniTensor
     * \p Lt is given as \f$ T_i \f$ and the template type \p rc is given as
     *  \f$ c \f$, then the result will be:
     * \f[
     *   T_o[i] = T_i[i] \div c,
     * \f]
     * where \f$ T_i[i] \f$ and \f$ T_o[i] \f$ are the elements in the
     * UniTensor \f$ T_i \f$ and \f$ T_o \f$.
     * @param[in] Lt The left UniTensor.
     * @param[in] rc
     * @parblock
     * The right template type.
     *
     * supported type:
     * Scalar, cytnx::cytnx_complex128, cytnx::cytnx_complex64,
     * cytnx::cytnx_double, cytnx::cytnx_float, cytnx::cytnx_int64,
     * cytnx::cytnx_uint64, cytnx::cytnx_int32, cytnx::cytnx_uint32,
     * cytnx::cytnx_int16, cytnx::cytnx_uint16, cytnx::cytnx_bool.
     * @endparblock
     * @return The result UniTensor.
     * @pre The supported template type shown above.
     * @note
     * 1. The inpute template type \p rc will be casted to the same type as
     * the UniTensor \p Lt.
     * 2. The division by zero is not allowed.
     * @see
     *    operator/(const UniTensor &Lt, const T &rc),
     *    Div(const cytnx::UniTensor &Lt, const T &rc)
     */
    template <class T>
    cytnx::UniTensor Div(const cytnx::UniTensor &Lt, const T &rc);

    // Mod:
    //==================================================
    /**
    @brief element-wise modulo
    */
    /**
     * @brief The modulo function between two UniTensor.
     * @details This is the modulo function for UniTensor. It will
     * perform the element-wise modulo between two UniTensor. That means if the
     * UniTensor \p Lt is given as \f$ T_L \f$ and the UniTensor \p Rt is given as
     * \f$ T_R \f$, then the result will be:
     * \f[
     *   T_o[i] = T_L[i] \mod T_R[i],
     * \f]
     * where \f$ T_L[i] \f$ and \f$ T_R[i] \f$ are the elements in the
     * UniTensor \f$ T_L \f$ and \f$ T_R \f$.
     * @param[in] Lt The left UniTensor.
     * @param[in] Rt The right UniTensor.
     * @return The result UniTensor.
     * @pre
     * 1. \p Lt and \p Rt must have the same shape.
     * 2. The input UniTensor \p Lt and \p Rt need to be integer type.
     * @see
     *   UniTensor::Mod(const UniTensor &Rt) const,
     *   operator%(const UniTensor &Lt, const UniTensor &Rt)
     */
    cytnx::UniTensor Mod(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt);

    /**
     * @brief The modulo function between a UniTensor and a template type.
     * @details This is the modulo function for UniTensor. It will
     * perform the element-wise modulo between a UniTensor and a template type.
     * That means if the template type \p lc is given as \f$ c \f$ and the
     * UniTensor \p Rt is given as \f$ T_i \f$, then the result will be:
     * \f[
     *  T_o[i] = c \mod T_i[i],
     * \f]
     * where \f$ T_i[i] \f$ and \f$ T_o[i] \f$ are the elements in the
     * UniTensor \f$ T_i \f$ and \f$ T_o \f$.
     * @param lc The left template type.
     * @parblock
     * The right template type.
     *
     * supported type:
     * Scalar, cytnx::cytnx_int64,
     * cytnx::cytnx_uint64, cytnx::cytnx_int32, cytnx::cytnx_uint32,
     * cytnx::cytnx_int16, cytnx::cytnx_uint16, cytnx::cytnx_bool.
     * @endparblock
     * @param Rt The right UniTensor.
     * @return The result UniTensor.
     * @pre The input \p lc and \p Rt need to be integer type.
     * @note
     * The inpute template type \p lc will be casted to the same type as
     * the UniTensor \p Rt.
     * @see
     *    operator%(const UniTensor &Lt, const T &rc),
     *    Mod(const cytnx::UniTensor &Lt, const T &rc)
     */
    template <class T>
    cytnx::UniTensor Mod(const T &lc, const cytnx::UniTensor &Rt);

    /**
     * @brief The modulo function between a UniTensor and a template type.
     * @details This is the modulo function for UniTensor. It will
     * perform the element-wise modulo between a UniTensor and a template type.
     * That means if the UniTensor \p Lt is given as \f$ T_i \f$ and the template
     * type \p rc is given as \f$ c \f$, then the result will be:
     * \f[
     *   T_o[i] = T_i[i] \mod c,
     * \f]
     * where \f$ T_i[i] \f$ and \f$ T_o[i] \f$ are the elements in the
     * UniTensor \f$ T_i \f$ and \f$ T_o \f$.
     * @param[in] Lt The left UniTensor.
     * @param[in] rc
     * @parblock
     * The right template type.
     *
     * supported type:
     * Scalar, cytnx::cytnx_int64,
     * cytnx::cytnx_uint64, cytnx::cytnx_int32, cytnx::cytnx_uint32,
     * cytnx::cytnx_int16, cytnx::cytnx_uint16, cytnx::cytnx_bool.
     * @endparblock
     * @return The result UniTensor.
     * @pre The input \p Lt and \p rc need to be integer type.
     * @note
     * The inpute template type \p rc will be casted to the same type as
     * the UniTensor \p Lt.
     * @see
     *    operator%(const UniTensor &Lt, const T &rc),
     *    Mod(const cytnx::UniTensor &Lt, const T &rc)
     */
    template <class T>
    cytnx::UniTensor Mod(const cytnx::UniTensor &Lt, const T &rc);

    /**
    @brief Perform Singular-Value decomposition on a UniTensor using divide-and-conquer method.
    @details This function performs the Singular-Value decomposition on a UniTensor \p Tin.
    The result will depend on the rowrank of the UniTensor \p Tin. For more details, please
    refer to the documentation of the function Svd(const Tensor &Tin, const bool &is_UvT).
    */
    std::vector<cytnx::UniTensor> Svd(const cytnx::UniTensor &Tin, const bool &is_UvT = true);

    /**
    @brief Perform Singular-Value decomposition on a UniTensor using ?gesvd method.
    @details This function performs the Singular-Value decomposition on a UniTensor \p Tin.
    The result will depend on the rowrank of the UniTensor \p Tin. For more details, please
    refer to the documentation of the function Gesvd(const Tensor &Tin, const bool &is_U, const bool
    &is_vT).
    */
    std::vector<cytnx::UniTensor> Gesvd(const cytnx::UniTensor &Tin, const bool &is_U = true,
                                        const bool &is_vT = true);

    /**
     * @brief Perform Singular-Value decomposition on a UniTensor with truncation.
     * @details This function performs the Singular-Value decomposition on a UniTensor \p Tin and
     * do the truncation on the singular values. The result will depend on the rowrank of the
     * UniTensor \p Tin. For more details, please refer to the documentation of the function
     * Svd_truncate(const Tensor &Tin, const cytnx_uint64 &keepdim, const double &err,
     *              const bool &is_UvT, const unsigned int &return_err).
     * @see Svd_truncate(const Tensor &Tin, const cytnx_uint64 &keepdim, const double &err,
     *                   const bool &is_UvT, const unsigned int &return_err)
     */
    std::vector<cytnx::UniTensor> Svd_truncate(const cytnx::UniTensor &Tin,
                                               const cytnx_uint64 &keepdim, const double &err = 0,
                                               const bool &is_UvT = true,
                                               const unsigned int &return_err = 0,
                                               const unsigned int &mindim = 0);

    /**
     * @brief Perform Singular-Value decomposition on a UniTensor with truncation.
     * @details This function performs the Singular-Value decomposition on a UniTensor \p Tin and
     * do the truncation on the singular values. The result will depend on the rowrank of the
     * UniTensor \p Tin. For more details, please refer to the documentation of the function
     * Gesvd_truncate(const Tensor &Tin, const cytnx_uint64 &keepdim, const double &err,
     *              const bool &is_U, const bool &is_vT, const unsigned int &return_err).
     * @see Gesvd_truncate(const Tensor &Tin, const cytnx_uint64 &keepdim, const double &err,
     *                   const bool &is_U, const bool &is_vT, const unsigned int &return_err)
     */
    std::vector<cytnx::UniTensor> Gesvd_truncate(const cytnx::UniTensor &Tin,
                                                 const cytnx_uint64 &keepdim, const double &err = 0,
                                                 const bool &is_U = true, const bool &is_vT = true,
                                                 const unsigned int &return_err = 0,
                                                 const unsigned int &mindim = 0);

    std::vector<cytnx::UniTensor> Hosvd(
      const cytnx::UniTensor &Tin, const std::vector<cytnx_uint64> &mode,
      const bool &is_core = true, const bool &is_Ls = false,
      const std::vector<cytnx_int64> &trucate_dim = std::vector<cytnx_int64>());

    /**
     * @brief Perform the exponential function on a UniTensor, which the blocks are
     * Hermitian matrix.
     * @details This function performs the exponential function on a UniTensor \p Tin, which the
     * blocks are Hermitian matrix. For more details, please refer to the documentation of the
     * function ExpH(const Tensor &Tin, const T &a, const T &b).
     * @see ExpH(const Tensor &Tin, const T &a, const T &b)
     */
    template <typename T>
    cytnx::UniTensor ExpH(const cytnx::UniTensor &Tin, const T &a, const T &b = 0);

    /**
     * @brief Perform the exponential function on a UniTensor.
     * @details This function performs the exponential function on a UniTensor \p Tin.
     * For more details, please refer to the documentation of the
     * function ExpM(const Tensor &Tin, const T &a, const T &b).
     * @see ExpM(const Tensor &Tin, const T &a, const T &b)
     */
    template <typename T>
    cytnx::UniTensor ExpM(const cytnx::UniTensor &Tin, const T &a, const T &b = 0);

    /**
     * @brief Perform the exponential function on a UniTensor, which the blocks are
     * Hermitian matrix.
     * @details This function performs the exponential function on a UniTensor \p Tin, which the
     * blocks are Hermitian matrix. For more details, please refer to the documentation of the
     * function ExpH(const Tensor &Tin)
     * @see ExpH(const Tensor &Tin)
     */
    cytnx::UniTensor ExpH(const cytnx::UniTensor &Tin);

    /**
     * @brief Perform the exponential function on a UniTensor.
     * @details This function performs the exponential function on a UniTensor \p Tin.
     * For more details, please refer to the documentation of the
     * function ExpM(const Tensor &Tin)
     * @see ExpM(const Tensor &Tin)
     */
    cytnx::UniTensor ExpM(const cytnx::UniTensor &Tin);

    /**
     * @deprecated This function is deprecated, please use
     * Trace(const cytnx::UniTensor &Tin, const string &a, const string &b) instead.
     */
    cytnx::UniTensor Trace(const cytnx::UniTensor &Tin, const cytnx_int64 &a = 0,
                           const cytnx_int64 &b = 1);

    /**
     * @brief Perform trace over two legs of a UniTensor.
     * @details This function performs trace over two legs of a UniTensor \p Tin. The two legs
     * are specified by \p a and \p b. For more details, please refer to the documentation of the
     * function Trace(const Tensor &Tin, const cytnx_int64 &a, const cytnx_int64 &b).
     * @see Trace(const Tensor &Tin, const cytnx_int64 &a, const cytnx_int64 &b)
     */
    cytnx::UniTensor Trace(const cytnx::UniTensor &Tin, const std::string &a, const std::string &b);

    /**
     * @brief Perform the QR decomposition on a UniTensor.
     * @details This function performs the QR decomposition on a UniTensor \p Tin.
     * The result will depend on the rowrank of the UniTensor \p Tin. For more details,
     * please refer to the documentation of the function
     * Qr(const Tensor &Tin, const bool &is_tau).
     * @see Qr(const Tensor &Tin, const bool &is_tau)
     */
    std::vector<cytnx::UniTensor> Qr(const cytnx::UniTensor &Tin, const bool &is_tau = false);

    /**
     * @brief Perform the QDR decomposition on a UniTensor.
     * @details This function performs the QDR decomposition on a UniTensor \p Tin.
     * The result will depend on the rowrank of the UniTensor \p Tin. For more details,
     * please refer to the documentation of the function
     * Qdr(const Tensor &Tin, const bool &is_tau).
     * @see Qdr(const Tensor &Tin, const bool &is_tau)
     */
    std::vector<cytnx::UniTensor> Qdr(const cytnx::UniTensor &Tin, const bool &is_tau = false);

    // Pow:
    //==================================================
    /**
    @brief take power p on all the elements in UniTensor.
    @details This function will take power p on all the elements in UniTensor.
    @param[in] Tin the input UniTensor
    @param[in] p the power
    @pre If \p Tin is a real UniTensor and containt negative elements,
    then \p p must be an integer.
    @return UniTensor with the same shape as Tin, but with the elements are the power of Tin.
    @note Compare to the Pow_(UniTensor &Tin, const double &p) function, this
    function will not modify the input UniTensor and return a new UniTensor.
    @see Pow_(UniTensor &Tin, const double &p)
    */
    UniTensor Pow(const UniTensor &Tin, const double &p);

    /**
     * @brief Take power p on all the elements in UniTensor, inplacely.
     * @details This function will take power p on all the elements in UniTensor, inplacely.
     * @param[in,out] Tin the input UniTensor
     * @param[in] p the power
     * @pre If \p Tin is a real UniTensor and containt negative elements,
     * then \p p must be an integer.
     * @note Compare to the Pow function, this is an inplacely function, which
     * will modify the input UniTensor.
     * @see Pow(const UniTensor &Tin, const double &p)
     */
    void Pow_(UniTensor &Tin, const double &p);

    /**
     * @brief Elementwise conjugate of the UniTensor
     * @param[in] UT The input UniTensor.
     * @return [UniTensor] The UniTensor with all element being conjugated
     * @see See UniTensor.Conj() for further details
     */
    cytnx::UniTensor Conj(const cytnx::UniTensor &UT);

    /**
     * @brief Inplace elementwise conjugate of the UniTensor
     * @param[in] UT The input UniTensor.
     * @see See UniTensor.Conj_() for further details
     */
    void Conj_(cytnx::UniTensor &UT);

    //====================================================================================
    //  [Tensor]
    //  ====================================================================================
    //====================================================================================

    /**
     * @bridf The addition function for Tensor.
     * @details This is the addtion function between two Tensor. It will perform
     * the element-wise addtion. That means if the left Tensor \p Lt
     * is given as \f$ T_L \f$ and the right Tensor \p Rt is given as \f$ T_R \f$,
     * then the result will be:
     *   \f[
     *    T_o[i] = T_L[i] + T_R[i],
     *  \f]
     * where \f$ T_L[i] \f$ and \f$ T_R[i] \f$ are the elements in the
     * Tensor \f$ T_L \f$ and \f$ T_R \f$.
     * It will perform the element-wise addtion and note that it will return a
     * new Tensor object.
     * @param[in] Lt The left Tensor.
     * @param[in] Rt The right Tensor.
     * @return The result Tensor.
     * @pre The shape of \p Lt and \p Rt must be the same.
     * @see
     * Add(const T &lc, const Tensor &Rt),
     * Add(const Tensor &Lt, const T &rc),
     * iAdd(Tensor &Lt, const Tensor &Rt),
     * operator+(const Tensor &Lt, const Tensor &Rt)
     */
    Tensor Add(const Tensor &Lt, const Tensor &Rt);

    /**
     * @brief The addition function for Tensor.
     * @details This is the addtion function between a Tensor and a template type.
     * It will perform the element-wise addtion. That means if the template type \p lc
     * is given as \f$ c \f$ and the Tensor \p Rt is given as \f$ T_R \f$,
     * then the result will be:
     *   \f[
     *    T_o[i] = c + T_L[i],
     *  \f]
     * where \f$ T_R[i] \f$ is the elements in the Tensor \f$ T_R \f$.
     * It will perform the element-wise addtion and note that it will return a
     * new Tensor object.
     * @param[in] lc The left template type.
     * @param[in] Rt The right Tensor.
     * @return The result Tensor.
     * @see
     * Add(const Tensor &Lt, const Tensor &Rt),
     * Add(const Tensor &Lt, const T &rc),
     * iAdd(Tensor &Lt, const Tensor &Rt),
     * operator+(const Tensor &Lt, const Tensor &Rt)
     */
    template <class T>
    Tensor Add(const T &lc, const Tensor &Rt);

    /**
     * @brief The addition function for Tensor.
     * @details This is the addtion function between a Tensor and a template type.
     * It will perform the element-wise addtion. That means if the Tensor \p Lt
     * is given as \f$ T_L \f$ and the template type \p rc is given as \f$ c \f$,
     * then the result will be:
     *   \f[
     *    T_o[i] = T_L[i] + c,
     *  \f]
     * where \f$ T_L[i] \f$ is the elements in the Tensor \f$ T_L \f$.
     * It will perform the element-wise addtion and note that it will return a
     * new Tensor object.
     * @param[in] Lt The left Tensor.
     * @param[in] rc The right template type.
     * @return The result Tensor.
     * @see
     * Add(const Tensor &Lt, const Tensor &Rt),
     * Add(const T &lc, const Tensor &Rt),
     * iAdd(Tensor &Lt, const Tensor &Rt),
     * operator+(const Tensor &Lt, const Tensor &Rt)
     */
    template <class T>
    Tensor Add(const Tensor &Lt, const T &rc);

    /**
     * @brief The addition function for Tensor, inplacely.
     * @details This is the inplace addtion function between two Tensor. It will perform
     * the element-wise addtion. That means if the left Tensor \p Lt
     * is given as \f$ T_L \f$ and the right Tensor \p Rt is given as \f$ T_R \f$,
     * then the result will be:
     *   \f[
     *    T_L[i] = T_L[i] + T_R[i],
     *  \f]
     * where \f$ T_L[i] \f$ and \f$ T_R[i] \f$ are the elements in the
     * Tensor \f$ T_L \f$ and \f$ T_R \f$.
     * It will perform the element-wise addtion and note that it will modify the
     * left Tensor \p Lt.
     * @param[in,out] Lt The left Tensor.
     * @param[in] Rt The right Tensor.
     * @pre The shape of \p Lt and \p Rt must be the same.
     * @note Compare to the function Add(const Tensor &Lt, const Tensor &Rt),
     * this is a inplace function and it will modify the left Tensor \p Lt.
     * @see
     * Add(const Tensor &Lt, const Tensor &Rt),
     * Add(const T &lc, const Tensor &Rt),
     * Add(const Tensor &Lt, const T &rc),
     * operator+(const Tensor &Lt, const Tensor &Rt)
     */
    void iAdd(Tensor &Lt, const Tensor &Rt);

    // Sub:
    //==================================================
    /**
     * @brief The subtraction function for Tensor.
     * @details This is the subtraction function between two Tensor. It will perform
     * the element-wise subtraction. That means if the left Tensor \p Lt
     * is given as \f$ T_L \f$ and the right Tensor \p Rt is given as \f$ T_R \f$,
     * then the result will be:
     *  \f[
     *   T_o[i] = T_L[i] - T_R[i],
     *  \f]
     * where \f$ T_L[i] \f$ and \f$ T_R[i] \f$ are the elements in the
     * Tensor \f$ T_L \f$ and \f$ T_R \f$.
     * It will perform the element-wise subtraction and note that it will return a
     * new Tensor object.
     * @param[in] Lt The left Tensor.
     * @param[in] Rt The right Tensor.
     * @return The result Tensor.
     * @see
     * Sub(const T &lc, const Tensor &Rt),
     * Sub(const Tensor &Lt, const T &rc),
     * iSub(Tensor &Lt, const Tensor &Rt),
     * operator-(const Tensor &Lt, const Tensor &Rt)
     */
    Tensor Sub(const Tensor &Lt, const Tensor &Rt);

    /**
     * @brief The subtraction function for Tensor.
     * @details This is the subtraction function between a Tensor and a template type.
     * It will perform the element-wise subtraction. That means if the template type
     * \p lc is given as \f$ c \f$ and the right Tensor \p Rt is given as \f$ T_R \f$,
     * then the result will be:
     * \f[
     * T_o[i] = c - T_R[i],
     * \f]
     * where \f$ T_R[i] \f$ is the elements in the Tensor \f$ T_R \f$.
     * It will perform the element-wise subtraction and note that it will return a
     * new Tensor object.
     * @param[in] lc The left template type.
     * @param[in] Rt The right Tensor.
     * @return The result Tensor.
     * @see
     * Sub(const Tensor &Lt, const Tensor &Rt),
     * Sub(const Tensor &Lt, const T &rc),
     * iSub(Tensor &Lt, const Tensor &Rt),
     * operator-(const Tensor &Lt, const Tensor &Rt)
     */
    template <class T>
    Tensor Sub(const T &lc, const Tensor &Rt);

    /**
     * @brief The subtraction function for Tensor.
     * @details This is the subtraction function between a Tensor and a template type.
     * It will perform the element-wise subtraction. That means if the left Tensor
     * \p Lt is given as \f$ T_L \f$ and the template type \p rc is given as \f$ c \f$,
     * then the result will be:
     * \f[
     * T_o[i] = T_L[i] - c,
     * \f]
     * where \f$ T_L[i] \f$ is the elements in the Tensor \f$ T_L \f$.
     * It will perform the element-wise subtraction and note that it will return a
     * new Tensor object.
     * @param[in] Lt The left Tensor.
     * @param[in] rc The right template type.
     * @return The result Tensor.
     * @see
     * Sub(const Tensor &Lt, const Tensor &Rt),
     * Sub(const T &lc, const Tensor &Rt),
     * iSub(Tensor &Lt, const Tensor &Rt),
     * operator-(const Tensor &Lt, const Tensor &Rt)
     */
    template <class T>
    Tensor Sub(const Tensor &Lt, const T &rc);

    /**
     * @brief The subtraction function for Tensot, inplscely.
     * @details This is the subtraction function between two Tensor. It will perform
     * the element-wise subtraction. That means if the left Tensor \p Lt
     * is given as \f$ T_L \f$ and the right Tensor \p Rt is given as \f$ T_R \f$,
     * then the result will be:
     * \f[
     * T_L[i] = T_L[i] - T_R[i],
     * \f]
     * where \f$ T_L[i] \f$ and \f$ T_R[i] \f$ are the elements in the
     * Tensor \f$ T_L \f$ and \f$ T_R \f$.
     * It will perform the element-wise subtraction and note that it will modify the
     * left Tensor \p Lt.
     * @param[in,out] Lt The left Tensor.
     * @param[in] Rt The right Tensor.
     * @pre The shape of \p Lt and \p Rt must be the same.
     * @note Compare to the function Sub(const Tensor &Lt, const Tensor &Rt),
     * this is a inplace function and it will modify the left Tensor \p Lt.
     * @see
     * Sub(const Tensor &Lt, const Tensor &Rt),
     * Sub(const T &lc, const Tensor &Rt),
     * Sub(const Tensor &Lt, const T &rc),
     * operator-(const Tensor &Lt, const Tensor &Rt)
     */
    void iSub(Tensor &Lt, const Tensor &Rt);

    // Mul:
    //==================================================
    /**
     * @brief The multiplication function for Tensor.
     * @details This is the multiplication function between two Tensor. It will perform
     * the element-wise multiplication. That means if the left Tensor \p Lt
     * is given as \f$ T_L \f$ and the right Tensor \p Rt is given as \f$ T_R \f$,
     * then the result will be:
     * \f[
     * T_o[i] = T_L[i] * T_R[i]
     * \f]
     * where \f$ T_L[i] \f$ and \f$ T_R[i] \f$ are the elements in the
     * Tensor \f$ T_L \f$ and \f$ T_R \f$.
     * It will perform the element-wise multiplication and note that it will return a
     * new Tensor object.
     * @param[in] Lt The left Tensor.
     * @param[in] Rt The right Tensor.
     * @return The result Tensor.
     * @see
     * Mul(const T &lc, const Tensor &Rt),
     * Mul(const Tensor &Lt, const T &rc),
     * iMul(Tensor &Lt, const Tensor &Rt),
     * operator*(const Tensor &Lt, const Tensor &Rt)
     */
    Tensor Mul(const Tensor &Lt, const Tensor &Rt);

    /**
     * @brief The multiplication function for Tensor.
     * @details This is the multiplication function between a Tensor and a template type.
     * It will perform the element-wise multiplication. That means if the left Tensor
     * \p Lt is given as \f$ T_L \f$ and the template type \p rc is given as \f$ c \f$,
     * then the result will be:
     * \f[
     * T_o[i] = T_L[i] * c,
     * \f]
     * where \f$ T_L[i] \f$ is the elements in the Tensor \f$ T_L \f$.
     * It will perform the element-wise multiplication and note that it will return a
     * new Tensor object.
     * @param[in] Lt The left Tensor.
     * @param[in] rc The right template type.
     * @return The result Tensor.
     * @see
     * Mul(const Tensor &Lt, const Tensor &Rt),
     * Mul(const T &lc, const Tensor &Rt),
     * iMul(Tensor &Lt, const Tensor &Rt),
     * operator*(const Tensor &Lt, const Tensor &Rt)
     */
    template <class T>
    Tensor Mul(const T &lc, const Tensor &Rt);

    /**
     * @brief The multiplication function for Tensor.
     * @details This is the multiplication function between a Tensor and a template type.
     * It will perform the element-wise multiplication. That means if the left Tensor
     * \p Lt is given as \f$ T_L \f$ and the template type \p rc is given as \f$ c \f$,
     * then the result will be:
     * \f[
     * T_o[i] = T_L[i] * c,
     * \f]
     * where \f$ T_L[i] \f$ is the elements in the Tensor \f$ T_L \f$.
     * It will perform the element-wise multiplication and note that it will return a
     * new Tensor object.
     * @param[in] Lt The left Tensor.
     * @param[in] rc The right template type.
     * @return The result Tensor.
     * @see
     * Mul(const Tensor &Lt, const Tensor &Rt),
     * Mul(const T &lc, const Tensor &Rt),
     * iMul(Tensor &Lt, const Tensor &Rt),
     * operator*(const Tensor &Lt, const Tensor &Rt)
     */
    template <class T>
    Tensor Mul(const Tensor &Lt, const T &rc);

    /**
     * @brief The multiplication function for Tensor, inplacely.
     * @details This is the multiplication function between two Tensor. It will perform
     * the element-wise multiplication. That means if the left Tensor \p Lt
     * is given as \f$ T_L \f$ and the right Tensor \p Rt is given as \f$ T_R \f$,
     * then the result will be:
     * \f[
     * T_o[i] = T_L[i] * T_R[i]
     * \f]
     * where \f$ T_L[i] \f$ and \f$ T_R[i] \f$ are the elements in the
     * Tensor \f$ T_L \f$ and \f$ T_R \f$.
     * It will perform the element-wise multiplication and note that it will modify the
     * left Tensor \p Lt.
     * @param[in,out] Lt The left Tensor.
     * @param[in] Rt The right Tensor.
     * @note
     * Compare to Mul(const Tensor &Lt, const Tensor &Rt), this is inplace function
     * and will modify the left Tensor \p Lt.
     * @see
     * Mul(const Tensor &Lt, const Tensor &Rt),
     * Mul(const T &lc, const Tensor &Rt),
     * Mul(const Tensor &Lt, const T &rc),
     * operator*(const Tensor &Lt, const Tensor &Rt)
     */
    void iMul(Tensor &Lt, const Tensor &Rt);

    // Div:
    //==================================================
    /**
     * @brief The division function for Tensor.
     * @details This is the division function between two Tensor. It will perform
     * the element-wise division. That means if the left Tensor \p Lt
     * is given as \f$ T_L \f$ and the right Tensor \p Rt is given as \f$ T_R \f$,
     * then the result will be:
     * \f[
     * T_o[i] = T_L[i] / T_R[i]
     * \f]
     * where \f$ T_L[i] \f$ and \f$ T_R[i] \f$ are the elements in the
     * Tensor \f$ T_L \f$ and \f$ T_R \f$.
     * It will perform the element-wise division and note that it will return a
     * new Tensor object.
     * @param[in] Lt The left Tensor.
     * @param[in] Rt The right Tensor.
     * @return The result Tensor.
     * @pre the right Tensor \p Rt should not contain any zero element.
     * @see
     * Div(const T &lc, const Tensor &Rt),
     * Div(const Tensor &Lt, const T &rc),
     * iDiv(Tensor &Lt, const Tensor &Rt),
     * operator/(const Tensor &Lt, const Tensor &Rt)
     */
    Tensor Div(const Tensor &Lt, const Tensor &Rt);

    /**
     * @brief The division function for Tensor.
     * @details This is the division function between a Tensor and a template type.
     * It will perform the element-wise division. That means if the left template type
     * \p lc is given as \f$ c \f$ and the right Tensor \p Rt is given as \f$ T_R \f$,
     * then the result will be:
     * \f[
     * T_o[i] = c / T_R[i]
     * \f]
     * where \f$ T_L[i] \f$ is the elements in the Tensor \f$ T_L \f$.
     * It will perform the element-wise division and note that it will return a
     * new Tensor object.
     * @param[in] lc The left template type.
     * @param[in] Rt The right Tensor.
     * @return The result Tensor.
     * @pre the right tensor \p Rt should not contain any zero element.
     * @see
     * Div(const Tensor &Lt, const Tensor &Rt),
     * Div(const Tensor &Lt, const T &rc),
     * iDiv(Tensor &Lt, const Tensor &Rt),
     * operator/(const Tensor &Lt, const Tensor &Rt)
     */
    template <class T>
    Tensor Div(const T &lc, const Tensor &Rt);

    /**
     * @brief The division function for Tensor.
     * @details This is the division function between a Tensor and a template type.
     * It will perform the element-wise division. That means if the left Tensor \p Lt
     * is given as \f$ T_L \f$ and the right template type \p rc is given as \f$ c \f$,
     * then the result will be:
     * \f[
     * T_o[i] = T_L[i] / c
     * \f]
     * where \f$ T_L[i] \f$ is the elements in the Tensor \f$ T_L \f$.
     * It will perform the element-wise division and note that it will return a
     * new Tensor object.
     * @param[in] Lt The left Tensor.
     * @param[in] rc The right template type.
     * @return The result Tensor.
     * @pre the right template type \p rc should not be zero.
     * @see
     * Div(const Tensor &Lt, const Tensor &Rt),
     * Div(const T &lc, const Tensor &Rt),
     * iDiv(Tensor &Lt, const Tensor &Rt),
     * operator/(const Tensor &Lt, const Tensor &Rt)
     */
    template <class T>
    Tensor Div(const Tensor &Lt, const T &rc);

    /**
     * @brief The inplace division function for Tensor, inplacely.
     * @details This is the inplace division function between two Tensor. It will perform
     * the element-wise division. That means if the left Tensor \p Lt
     * is given as \f$ T_L \f$ and the right Tensor \p Rt is given as \f$ T_R \f$,
     * then the result will be:
     * \f[
     * T_o[i] = T_L[i] / T_R[i]
     * \f]
     * where \f$ T_L[i] \f$ and \f$ T_R[i] \f$ are the elements in the
     * Tensor \f$ T_L \f$ and \f$ T_R \f$.
     * It will perform the element-wise division and note that it will modify the
     * left Tensor \p Lt.
     * @param[in,out] Lt The left Tensor.
     * @param[in] Rt The right Tensor.
     * @pre the right Tensor \p Rt should not contain any zero element.
     * @note compare to the Div(const Tensor &Lt, const Tensor &Rt) function,
     * this is a inplace function, which will modify the left Tensor \p Lt.
     * @see
     * Div(const Tensor &Lt, const Tensor &Rt),
     * Div(const T &lc, const Tensor &Rt),
     * Div(const Tensor &Lt, const T &rc),
     * operator/(const Tensor &Lt, const Tensor &Rt)
     */
    void iDiv(Tensor &Lt, const Tensor &Rt);

    // Mod:
    //==================================================

    /**
     * @brief The mod function for Tensor.
     * @details This is the mod function between two Tensor. It will perform
     * the element-wise mod. That means if the left Tensor \p Lt
     * is given as \f$ T_L \f$ and the right Tensor \p Rt is given as \f$ T_R \f$,
     * then the result will be:
     * \f[
     * T_o[i] = T_L[i] % T_R[i]
     * \f]
     * where \f$ T_L[i] \f$ and \f$ T_R[i] \f$ are the elements in the
     * Tensor \f$ T_L \f$ and \f$ T_R \f$.
     * It will perform the element-wise mod and note that it will return a
     * new Tensor object.
     * @param[in] Lt The left Tensor.
     * @param[in] Rt The right Tensor.
     * @return The result Tensor.
     * @pre The input tensors \p Lt and \p Rt should have the same shape and
     * need to be integer type.
     * @see
     * Mod(const T &lc, const Tensor &Rt),
     * Mod(const Tensor &Lt, const T &rc),
     */
    Tensor Mod(const Tensor &Lt, const Tensor &Rt);

    /**
     * @brief The mod function for Tensor.
     * @details This is the mod function between a Tensor and a template type.
     * It will perform the element-wise mod. That means if the left template
     * type \p lc is given as \f$ c \f$ and the right Tensor \p Rt is given as
     * \f$ T_R \f$, then the result will be:
     * \f[
     * T_o[i] = c % T_R[i]
     * \f]
     * where \f$ T_R[i] \f$ is the elements in the Tensor \f$ T_R \f$.
     * It will perform the element-wise mod and note that it will return a
     * new Tensor object.
     * @param[in] lc The left template type.
     * @param[in] Rt The right Tensor.
     * @return The result Tensor.
     * @pre the right template type \p rc should be integer type.
     * @see
     * Mod(const Tensor &Lt, const Tensor &Rt),
     * Mod(const Tensor &Lt, const T &rc)
     */
    template <class T>
    Tensor Mod(const T &lc, const Tensor &Rt);

    /**
     * @brief The mod function for Tensor.
     * @details This is the mod function between a Tensor and a template type.
     * It will perform the element-wise mod. That means if the left Tensor
     * \p Lt is given as \f$ T_L \f$ and the right template type \p rc is given as
     * \f$ c \f$, then the result will be:
     * \f[
     * T_o[i] = T_L[i] % c
     * \f]
     * where \f$ T_L[i] \f$ is the elements in the Tensor \f$ T_L \f$.
     * It will perform the element-wise mod and note that it will return a
     * new Tensor object.
     * @param[in] Lt The left Tensor.
     * @param[in] rc The right template type.
     * @return The result Tensor.
     * @pre the right template type \p rc should be integer type.
     * @see
     * Mod(const Tensor &Lt, const Tensor &Rt),
     * Mod(const T &lc, const Tensor &Rt)
     */
    template <class T>
    Tensor Mod(const Tensor &Lt, const T &rc);

    // Cpr:
    //==================================================
    /**
     * @brief The comparison function for Tensor.
     * @details This is the comparison function between two Tensor. It will
     * perform the element-wise comparison. That means if the left Tensor \p Lt
     * is given as \f$ T_L \f$ and the right Tensor \p Rt is given as \f$ T_R \f$,
     * then the result will be:
     * \f[
     * T_o[i] = \left\{\begin{array}{ll}
        1, & \text{if  } T_L[i]=T_R[i]\\
        0, & \text{else}
        \end{array}\right.
     * \f]
     * where \f$ T_o[i]\f$, \f$ T_L[i] \f$ and \f$ T_R[i] \f$ are the elements in the
     * Tensor \f$ T_o\f$, \f$ T_L \f$ and \f$ T_R \f$ and \f$ T_o[i] \f$ is the
     * output Tensor which is a boolean type (see cytnx::Type).
     * It will perform the element-wise comparison and note that it will return a
     * new Tensor object.
     * @param[in] Lt The left Tensor.
     * @param[in] Rt The right Tensor.
     * @return The result Tensor.
     * @pre The input tensors \p Lt and \p Rt should have the same shape.
     * @see
     * Cpr(const T &lc, const Tensor &Rt),
     * Cpr(const Tensor &Lt, const T &rc)
     */
    Tensor Cpr(const Tensor &Lt, const Tensor &Rt);

    /**
     * @brief The comparison function for Tensor.
     * @details This is the comparison function between a Tensor and a template type.
     * It will perform the element-wise comparison. That means if the left template
     * type \p lc is given as \f$ c \f$ and the right Tensor \p Rt is given as
     * \f$ T_R \f$, then the result will be:
     * \f[
     * T_o[i] = \left\{\begin{array}{ll}
        1, & \text{if  } c=T_R[i]\\
        0, & \text{else}
        \end{array}\right.
     * \f]
     * where \f$ T_o[i]\f$, \f$ c \f$ and \f$ T_R[i] \f$ are the elements in the
     * Tensor \f$ T_o\f$, \f$ c \f$ and \f$ T_R \f$ and \f$ T_o[i] \f$ is the
     * output Tensor which is a boolean type (see cytnx::Type).
     * It will perform the element-wise comparison and note that it will return a
     * new Tensor object.
     * @param[in] lc The left template type.
     * @param[in] Rt The right Tensor.
     * @return The result Tensor.
     * @see
     * Cpr(const Tensor &Lt, const Tensor &Rt),
     * Cpr(const Tensor &Lt, const T &rc)
     */
    template <class T>
    Tensor Cpr(const T &lc, const Tensor &Rt);

    /**
     * @brief The comparison function for Tensor.
     * @details This is the comparison function between a Tensor and a template type.
     * It will perform the element-wise comparison. That means if the left Tensor
     * \p Lt is given as \f$ T_L \f$ and the right template type \p rc is given as
     * \f$ c \f$, then the result will be:
     * \f[
     * T_o[i] = \left\{\begin{array}{ll}
        1, & \text{if  } T_L[i]=c\\
        0, & \text{else}
        \end{array}\right.
     * \f]
     * where \f$ T_o[i]\f$, \f$ T_L[i] \f$ and \f$ c \f$ are the elements in the
     * Tensor \f$ T_o\f$, \f$ T_L \f$ and \f$ c \f$ and \f$ T_o[i] \f$ is the
     * output Tensor which is a boolean type (see cytnx::Type).
     * It will perform the element-wise comparison and note that it will return a
     * new Tensor object.
     * @param[in] Lt The left Tensor.
     * @param[in] rc The right template type.
     * @return The result Tensor.
     * @see
     * Cpr(const Tensor &Lt, const Tensor &Rt),
     * Cpr(const T &lc, const Tensor &Rt)
     */
    template <class T>
    Tensor Cpr(const Tensor &Lt, const T &rc);

    // Norm:
    //=================================================
    /**
    @brief Calculate the norm of a tensor.
    @details This function will calculate the norm of a tensor. If the input tensor is rank-1,
    then the frobenius norm will be calculated. If the input tensor is rank-N with N>=2, then
    the tensor will be flatten (see @ref Tensor::flatten() const "flatten()") to 1d first,
    and then calculate the frobenius norm.
    @param[in] Tl input Tensor
    @return Tensor
    */
    Tensor Norm(const Tensor &Tl);

    // Norm:
    //=================================================
    /**
    @brief Calculate the norm of an UniTensor.
    @details This function will calculate the norm of an UniTensor. If the input UniTensor is
    rank-1, then the frobenius norm will be calculated. If the input UniTensor is rank-N with N>=2,
    then the blocks will be flatten (see @ref Tensor::flatten() const "flatten()") to 1d first, and
    then calculate the frobenius norm.
    @param[in] uTl input UniTensor
    @return Tensor
    */
    Tensor Norm(const UniTensor &uTl);

    // Det:
    //=================================================
    /**
    @brief Calculate the determinant of a tensor.
    @details This function will calculate the determinant of a <em> square matrix </em>.
    That means the input tensor should be a rank-2 tensor with shape (N,N).
    @param[in] Tl input a Tensor with shape (N,N)
    @return Tensor
    @pre the input tensor should be a rank-2 tensor with shape (N,N). (a square matrix)
    */
    Tensor Det(const Tensor &Tl);

    // Svd:
    //==================================================
    /**
    @brief Perform Singular-Value decomposition on a rank-2 Tensor (a @em matrix).
    @details This function will perform Singular-Value decomposition on a matrix (a rank-2
    Tensor). That means givent a matrix \p Tin as \f$ M \f$, then the result will be:
    \f[
    M = U S V^\dagger,
    \f]
    where \f$ U \f$ is a left uniform matrix, \f$ S \f$ is a diagonal matrix with singular
    values, and \f$ V^\dagger \f$ is the conjugate transpose of the right uniform matrix \f$ V
    \f$. Furthermore, \f$ U \f$ and \f$ V \f$ are unitary matrices, and \f$ S \f$ is a
    non-negative diagonal matrix.

    @param[in] Tin a Tensor, it should be a rank-2 tensor (matrix)
    @param[in] is_UvT whether need to return a left unitary matrix.
    @return
    @parblock
    [std::vector<Tensors>]

    1. The first tensor is a 1-d tensor contanin the singular values
    2. If \p is_UvT is true, then the tensors \f$ U,V^\dagger \f$ will be pushed back to the vector.
    @endparblock
    @pre The input tensor should be a rank-2 tensor (matrix).
    @see \ref Svd_truncate(const Tensor &Tin, const cytnx_uint64 &keepdim, const double &err, const
    bool &is_UvT, const unsigned int &return_err) "Svd_truncate"
    */
    std::vector<Tensor> Svd(const Tensor &Tin, const bool &is_UvT = true);

    // Gesvd:
    //==================================================
    /**
    @brief Perform Singular-Value decomposition on a rank-2 Tensor (a @em matrix).
    @details This function will perform Singular-Value decomposition on a matrix (a rank-2
    Tensor). That means givent a matrix \p Tin as \f$ M \f$, then the result will be:
    \f[
    M = U S V^\dagger,
    \f]
    where \f$ U \f$ is a left uniform matrix, \f$ S \f$ is a diagonal matrix with singular
    values, and \f$ V^\dagger \f$ is the conjugate transpose of the right uniform matrix \f$ V
    \f$. Furthermore, \f$ U \f$ and \f$ V \f$ are unitary matrices, and \f$ S \f$ is a
    non-negative diagonal matrix.

    @param[in] Tin a Tensor, it should be a rank-2 tensor (matrix)
    @param[in] is_U whether need to return left unitary matrix.
    @param[in] is_vT whether need to return right unitary matrix
    @return
    @parblock
    [std::vector<Tensors>]

    1. The first tensor is a 1-d tensor contanin the singular values
    2. If \p is_U is true, then the tensor \f$ U \f$ will be pushed back to the vector, and if \p
    is_vT is true, \f$ V^\dagger \f$ will be pushed back to the vector.
    @endparblock
    @pre The input tensor should be a rank-2 tensor (matrix).
    @see \ref Gesvd_truncate(const Tensor &Tin, const cytnx_uint64 &keepdim, const double &err,
    const bool &is_U, const bool &is_vT, const unsigned int &return_err) "Gesvd_truncate"
    */
    std::vector<Tensor> Gesvd(const Tensor &Tin, const bool &is_U = true, const bool &is_vT = true);

    // Svd_truncate:
    //==================================================
    /**
    @brief Perform the truncate Singular-Value decomposition on a rank-2 Tensor (a @em matrix).
    @details This function will perform the truncate Singular-Value decomposition
    on a matrix (a rank-2 Tensor). It will perform the SVD first, and then truncate the
    singular values to the given cutoff \p err. That means givent a matrix \p Tin as \f$ M \f$,
    then the result will be:
    \f[
    M = U S V^\dagger,
    \f]
    where \f$ S \f$ is a singular values matrix with the singular values truncated to the
    given cutoff \p err. The dimension of \f$ S \f$ is at most \p keepdim.

    @param[in] Tin a Tensor, it should be a rank-2 tensor (matrix)
    @param[in] keepdim the number (at most) of singular values to keep.
    @param[in] err the cutoff error (the singular values smaller than \p err will be truncated.)
    @param[in] is_UvT whether need to return a left unitary matrix and a right unitary matrix.
    @param[in] return_err whether need to return the error. If \p return_err is \em true, then
    largest error will be pushed back to the vector (The smallest singular value in the return
    singular values matrix \f$ S \f$.) If \p return_err is \em positive int, then it will return the
    full list of truncated singular values.
    @return
    @parblock
    [std::vector<Tensors>]

    1. The first tensor is a 1-d tensor contanin the singular values
    2. If \p is_UvT is true, then the tensor \f$ U,V^\dagger \f$ will be pushed back to the vector.
    4. If \p return_err is true, then the error will be pushed back to the vector.
    @endparblock
    @pre The input tensor should be a rank-2 tensor (matrix).
    @see \ref Svd(const Tensor &Tin, const bool &is_U, const bool &is_vT) "Svd"
    */
    std::vector<Tensor> Svd_truncate(const Tensor &Tin, const cytnx_uint64 &keepdim,
                                     const double &err = 0, const bool &is_UvT = true,
                                     const unsigned int &return_err = 0,
                                     const unsigned int &mindim = 0);

    // Gesvd_truncate:
    //==================================================
    /**
    @brief Perform the truncate Singular-Value decomposition on a rank-2 Tensor (a @em matrix).
    @details This function will perform the truncate Singular-Value decomposition
    on a matrix (a rank-2 Tensor). It will perform the SVD first, and then truncate the
    singular values to the given cutoff \p err. That means givent a matrix \p Tin as \f$ M \f$,
    then the result will be:
    \f[
    M = U S V^\dagger,
    \f]
    where \f$ S \f$ is a singular values matrix with the singular values truncated to the
    given cutoff \p err. The dimension of \f$ S \f$ is at most \p keepdim.

    @param[in] Tin a Tensor, it should be a rank-2 tensor (matrix)
    @param[in] keepdim the number (at most) of singular values to keep.
    @param[in] err the cutoff error (the singular values smaller than \p err will be truncated.)
    @param[in] is_U whether need to return a left unitary matrix.
    @param[in] is_vT whether need to return a right unitary matrix.
    @param[in] return_err whether need to return the error. If \p return_err is \em true, then
    largest error will be pushed back to the vector (The smallest singular value in the return
    singular values matrix \f$ S \f$.) If \p return_err is \em positive int, then it will return the
    full list of truncated singular values.
    @return
    @parblock
    [std::vector<Tensors>]

    1. The first tensor is a 1-d tensor contanin the singular values
    2. If \p is_U is true, then the tensor \f$ U\f$ will be pushed back to the vector.
    3. If \p is_U is true, then the tensor \f$ V^\dagger \f$ will be pushed back to the vector.
    4. If \p return_err is true, then the error will be pushed back to the vector.
    @endparblock
    @pre The input tensor should be a rank-2 tensor (matrix).
    @see \ref Svd(const Tensor &Tin, const bool &is_U, const bool &is_vT) "Svd"
    */
    std::vector<Tensor> Gesvd_truncate(const Tensor &Tin, const cytnx_uint64 &keepdim,
                                       const double &err = 0, const bool &is_U = true,
                                       const bool &is_vT = true, const unsigned int &return_err = 0,
                                       const unsigned int &mindim = 0);

    // Hosvd:
    std::vector<Tensor> Hosvd(
      const Tensor &Tin, const std::vector<cytnx_uint64> &mode, const bool &is_core = true,
      const bool &is_Ls = false,
      const std::vector<cytnx_int64> &trucate_dim = std::vector<cytnx_int64>());

    // Qr:
    //==================================================
    /**
    @brief Perform QR decomposition on a rank-2 Tensor.
    @details This function will perform QR decomposition on a matrix (a rank-2 Tensor). That means
    givent a matrix \p Tin as \f$ M \f$, then the result will be:
    \f[
    M = Q R,
    \f]
    where \f$ Q \f$ is a orthogonal matrix, and \f$ R \f$ is a right-upper triangular matrix.

    @param[in] Tin a Tensor, it should be a rank-2 tensor (a @em matrix)
    @param[in] is_tau if return the tau that contains the Householder reflectors that generate q
    along with r. The tau array contains scaling factors for the reflectors

    @return
    @parblock
    [std::vector<Tensors>]

    1. the first tensor is the orthomormal matrix \f$ Q \f$, a 2-d tensor (matrix)
    2. the second tensor is the right-upper triangular matrix \f$ R \f$, a 2-d tensor (matrix)
    3. the third tensor is the Householder reflectors \f$ H \f$, a 1-d tensor (vector).
    This tensor will only return when \p is_tau = @em true.
    @endparblock
    @pre The input tensor should be a rank-2 tensor (matrix).
    @see \ref Qdr(const Tensor &Tin, const bool &is_tau) "Qdr"
    */
    std::vector<Tensor> Qr(const Tensor &Tin, const bool &is_tau = false);

    // Qdr:
    //==================================================
    /**
    @brief Perform QDR decomposition on a rank-2 Tensor.
    @param[in] Tin a cytnx::Tensor, it should be a rank-2 tensor (matrix)
    @param[in] is_tau if return the tau that contains the Householder reflectors that generate q
    along with r. The tau array contains scaling factors for the reflectors

    @return
    @parblock
    [std::vector<Tensors>]

    1. the first tensor is the orthomormal matrix \f$ Q \f$, a 2-d tensor (matrix)
    2. the second tensor is the diagonal matrix \f$ D \f$, a 1-d tensor (diagonal matrix)
    3. the third tensor is the right-upper triangular matrix \f$ R \f$, a 2-d tensor (matrix)
    4. the forth tensor is the Householder reflectors \f$ H \f$, a 1-d tensor (vector).
    This tensor will only return when \p is_tau = @em true.
    @endparblock
    @pre The input tensor should be a rank-2 tensor (matrix).
    @see \ref Qr(const Tensor &Tin, const bool &is_tau) "Qr"
    */
    std::vector<Tensor> Qdr(const Tensor &Tin, const bool &is_tau = false);

    // Eigh:
    //==================================================
    /**
    @brief eigen-value decomposition for Hermitian matrix
    @details This function will perform eigen-value decomposition for Hermitian matrix.
    Given a matrix \p Tin as \f$ M \f$, then the result will be:
    \f[
    M = V D V^\dagger,
    \f]
    where \f$ V \f$ is a unitary matrix contains the eigen vectors,
    and \f$ D \f$ is a diagonal matrix contains the eigen values.

    @param[in] Tin a \link cytnx::Tensor Tensor \endlink, it should be a rank-2 tensor (matrix)
    @param[in] is_V whether need to return the eigen vectors.
    @param[in] row_V if set to ture, the return eigen vectors will be row form.
    @return [std::vector<Tensors>], the first tensor is the eigen values, a 1-d tensor (vector).
    The second tensor is the eigen vectors, a 2-d tensor (matrix). This tensor will only return when
    \p is_V = @em true. Furthermore, if \p row_V = @em true, then the eigen vectors will be row
    form. Otherwise, the eigen vectors will be column form.
    @pre the \p Tin should be a Hermitian matrix.
    @warning If \p Tin is not a Hermitian matrix, only the lower triangular matrix will be used.
    (This is strongly not recommended, please use
    Eig(const Tensor &Tin, const bool &is_V, const bool &row_v) instead.
    */
    std::vector<Tensor> Eigh(const Tensor &Tin, const bool &is_V = true, const bool &row_v = false);

    std::vector<UniTensor> Eigh(const UniTensor &Tin, const bool &is_V = true,
                                const bool &row_v = false);

    // Eig:
    //==================================================
    /**
    @brief eigen-value decomposition for generic square matrix
    @details This function will perform eigen-value decomposition for generic square matrix.
    Given a matrix \p Tin as \f$ M \f$, then the result will be:
    \f[
    M = V D V^{-1},
    \f]
    where \f$ V \f$ is a invertible matrix contains the eigen vectors,
    and \f$ D \f$ is a diagonal matrix contains the eigen values.

    @param[in] Tin A square matrix (a rank-2 Tensor)
    @param[in] is_V whether need to return the eigen vectors.
    @param[in] row_V if set to ture, the return eigen vectors will be row form.
    @return [std::vector<Tensors>], the first tensor is the eigen values, a 1-d tensor (vector).
    The second tensor is the eigen vectors, a 2-d tensor (matrix). This tensor will only return when
    \p is_V = @em true. Furthermore, if \p row_V = @em true, then the eigen vectors will be row
    form. Otherwise, the eigen vectors will be column form.
    @pre the \p Tin should be a square matrix.
    */
    std::vector<Tensor> Eig(const Tensor &Tin, const bool &is_V = true, const bool &row_v = false);

    std::vector<UniTensor> Eig(const UniTensor &Tin, const bool &is_V = true,
                               const bool &row_v = false);

    // Trace:
    //==================================================
    /**
    @brief perform trace over index.
    @details This function will perform trace over index \p axisA and \p axisB.
    For example, if \p Tn is a rank-4 tensor \f$ T \f$, then \p axisA = 0 and
    \p axisB = 2, then the result will be:
    \f[
    \mathrm{Tr}_{j,l}(T) = \sum_{i,k} T_{i,j,k,l}
    \f]
    @param[in] Tn a Tensor
    @param[in] axisA the first index to perform trace.
    @param[in] axisB the second index to perform trace.
    @pre the \p Tn should be at-least rank-2 Tensor.
    */
    Tensor Trace(const Tensor &Tn, const cytnx_uint64 &axisA = 0, const cytnx_uint64 &axisB = 1);

    // Min:
    //==================================================
    /**
    @brief get the minimum element.
    @param[in] Tn a cytnx::Tensor
    @note For complex TN, only real part is compared.
    */
    Tensor Min(const Tensor &Tn);

    // Max:
    //==================================================
    /**
    @brief get the maximum element.
    @param[in] Tn a cytnx::Tensor
    @note For complex TN, only real part is compared.
    */
    Tensor Max(const Tensor &Tn);

    // Sum:
    //==================================================
    /**
    @brief get the sum of all the elements.
    @param[in] Tn a cytnx::Tensor
    */
    Tensor Sum(const Tensor &Tn);

    // Matmul:
    //==================================================
    /**
    @brief perform matrix multiplication on two tensors.
    @details This function will perform matrix multiplication on two matrices
    (2-rank Tensor) \p TL and \p TR. The result will be:
    \f[
    T = T_L T_R
    \f]
    @param[in] TL a left Tensor
    @param[in] TR a right Tensor
    @pre the \p TL and \p TR should be rank-2 Tensor.
    */
    Tensor Matmul(const Tensor &TL, const Tensor &TR);

    // Matmul_dg:
    //==================================================
    /**
    @brief perform matrix multiplication on two Tensors with one rank-1 and the other rank-2 where
    the rank-1 represent the diagonal elements of the specific tensor.

    @note one of \p TL and \p TR should be rank-1 Tensor and the other should be rank-2 Tensor.
    */
    Tensor Matmul_dg(const Tensor &Tl, const Tensor &Tr);

    // InvM:
    //==================================================
    /**
    @brief Matrix inverse.
    @details This function will perform matrix inverse on the input matrix \p Tin.
    @return [Tensor] the inversion of the input matrix.
    @pre \p Tin should be a rank-2 Tensor.
    */
    Tensor InvM(const Tensor &Tin);
    UniTensor InvM(const UniTensor &Tin);
    /**
    @brief inplace matrix inverse.
    @details This function will perform matrix inverse on the input matrix \p Tin, inplacely.
    @note Compare to InvM, this is inlpace function. The input matrix will be modified to it's
    inverse.
    @pre the Tin should be a rank-2 Tensor.
    */
    void InvM_(Tensor &Tin);
    void InvM_(UniTensor &Tin);
    // Inv:
    //==================================================
    /**
    @brief Element-wise inverse with clip.
    @details This function will perform Element-wise inverse with clip. If
    A[i] < \p clip, then 1/A[i] = 0 will be set. That is, the out put will be:
    \f[
    A_{out} = \left\{
    \begin{array}{ll}
    1/A[i] & \mathrm{if} \ A[i] \geq \mathrm{clip} \\
    0 & \mathrm{otherwise}
    \end{array}
    \right.
    \f]
    @param[in] Tin a Tensor
    @param[in] clip the clip value.
    @return
        [Tensor]
    @note For complex type Tensors, the square norm is used to determine the clip.
    */
    Tensor Inv(const Tensor &Tin, const double &clip);

    /**
    @brief inplace perform Element-wise inverse with clip.
    @details This function will perform Element-wise inverse with clip. This function
    is just as same as Inv, but it will modify the input Tensor inplace.
    @param[in] Tin a Tensor
    @param[in] clip the clip value.
    @return [Tensor]
    @note
    1. For complex type Tensors, the square norm is used to determine the clip.
    2. on return, all the elements will be modified to it's inverse. if Tin is integer type, it
    will automatically promote to Type.Double.
    */
    void Inv_(Tensor &Tin, const double &clip);

    // Conj:
    //==================================================
    /**
    @brief Conjugate all the element in Tensor.
    @details This function take the complex conjugate of all the elements in Tensor \p Tin.
    That is, the output will be:
    \f[
        T_{o}[i] = T_{in}[i]^*
    \f]
    Furthermore,
        1. if the input Tensor is complex, then return a new Tensor with all the elements
         are conjugated.
        2. if the input Tensor is real, then return a copy of input Tensor.
    @param[in] Tin a Tensor
    @return [Tensor]
    */
    Tensor Conj(const Tensor &Tin);

    /**
    @brief inplace perform Conjugate on all the element in Tensor.
    @details This function take the complex conjugate of all the elements in Tensor \p Tin.
    This function is just a inplace version of Conj.
    @note
    1. if the input Tensor is complex, the elements of input Tensor will all be conjugated.
    2. if the input Tensor is real, then nothing act.
    */
    void Conj_(Tensor &Tin);

    // Exp:
    //==================================================
    /**
    @brief Exponential all the element in Tensor.
    @details This function will perform Exponential on all the elements in Tensor \p Tin.
    That is, the output will be:
    \f[
        T_{o}[i] = e^{T_{i}[i]}
    \f]
    Note that it will cast to Double type or ComplexDouble type.
    @param[in] Tin a Tensor
    @return
        [Double Tensor] or [ComplexDouble Tensor]

    */
    Tensor Exp(const Tensor &Tin);

    /**
    @brief Exponential all the element in Tensor.
    @details This function will perform Exponential on all the elements in Tensor \p Tin.
    That is, the output will be:
    \f[
        T_{o}[i] = e^{T_{i}[i]}
    \f]
    Note that it will cast to Float type or ComplexFloat type.
    @param[in] Tin a Tensor
    @return
        [Float Tensor] or [ComplexFloat Tensor]

    */
    Tensor Expf(const Tensor &Tin);

    /**
    @brief inplace perform Exponential on all the element in Tensor.
    @details This function will perform Exponential on all the elements in Tensor \p Tin.
    Furthermore,
        1. on return, the elements in Tin will be modified to it's exponetial value.
        2. For Real, if the type is not Double, change the type of the input tensor to Double.
        3. For Complex, if input is ComplexFloat, promote to ComplexDouble.
    @param[in] Tin, the input Tensor.
    */
    void Exp_(Tensor &Tin);

    /**
    @brief inplace perform Exponential on all the element in Tensor.
    @details This function will perform Exponential on all the elements in Tensor \p Tin.
    Furthermore,
        1. on return, the elements in Tin will be modified to it's exponetial value.
        2. For Real, if the type is not Float, change the type of the input tensor to Float.
        3. For Complex, if input is ComplexDouble, promote to ComplexFloat.
    @param[in] Tin, the input Tensor.
    */
    void Expf_(Tensor &Tin);

    // Pow:
    //==================================================
    /**
    @brief take power p on all the elements in Tensor.
    @details This function will perform power p on all the elements in Tensor \p Tin.
    That is, the output will be:
    \f[
        T_{o}[i] = T_{i}[i]^{p}
    \f]
    @param[in] p, the power
    @return [Tensor]

    */
    Tensor Pow(const Tensor &Tin, const double &p);

    /**
    @brief inplace perform power on all the elements in Tensor.
    @details this is just a inplace version of Pow. The input Tensor \p Tin will be modified.
    @param[in] Tin, the input Tensor.
    @param[in] p, the power.
    */
    void Pow_(Tensor &Tin, const double &p);

    // Abs:
    //==================================================
    /**
    @brief Elementwise absolute value.
    @details This function will perform Elementwise absolute value on all the elements in Tensor \p
    Tin. That is, the output will be: \f[ T_{o}[i] = |T_{i}[i]| \f]
    @param[in] Tin tensor.
    @return [Tensor]
    */
    Tensor Abs(const Tensor &Tin);

    /**
    @brief inplace perform elementwiase absolute value.
    @This is just a inplace version of Abs. The input Tensor \p Tin will be modified.
    @param[in] Tin, the input Tensor.
    @note on return, the elements in Tin will be modified to it's absolute value. Note that if the
    input tensor is complex, it will be modified to real type.
    */
    void Abs_(Tensor &Tin);

    // Diag:
    //==================================================
    /**
    @brief return a diagonal tensor with diagonal elements provided as Tin.
    @return
        [Tensor]
    @details This function will return a diagonal tensor with diagonal elements provided as \p Tin.
    Furthermore, the return Tensor will be rank-2, with shape=(L, L); where L is the number of
    elements in \p Tin. \pre \p Tin should be a rank-2 Tensor.
    */
    Tensor Diag(const Tensor &Tin);

    // Tensordot:
    //==================================================
    /**
    @brief perform tensor dot by sum out the indices assigned of two Tensors.
    @param[in] Tl Tensor #1
    @param[in] Tr Tensor #2
    @param[in] idxl the indices of rank of Tensor #1 that is going to sum with Tensor #2
    @param[in] idxr the indices of rank of Tensor #2 that is going to sum with Tensor #1
    @param[in] cacheL cache Tensor #1 (See user-guide for details)
    @param[in] cacheR cache Tensor #2 (See user-guide for details)
    @return
        [Tensor]

    @note
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
    @param[in] Tl Tensor #1
    @param[in] Tr Tensor #2
    @param[in] idxl the indices of rank of Tensor #1 that is going to sum with Tensor #2
    @param[in] idxr the indices of rank of Tensor #2 that is going to sum with Tensor #1
    @param[in] diag_L if Tl(true)/Tr(false) is a diagnal matrix, represented by a rank-1 tensor.
    @return
        [Tensor]

    @note
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
    @details This function will perform outer produces of two rank-1 Tensor \p Tl and \p Tr.
    Furthermore, if the Tensor \p Tl has [shape_1], and Tensor \p Tr has [shape_2]; then the return
    Tensor will have shape: concate(shape_1,shape_2)
    @param[in] Tl rank-1 Tensor #1
    @param[in] Tr rank-1 Tensor #2
    @return [Tensor]
    @pre two tensor should on same device.
    */
    Tensor Outer(const Tensor &Tl, const Tensor &Tr);

    // Kron:
    //==================================================
    /**
    @brief perform kronecker produces of two Tensor.
    @details This function will perform kronecker produces of two Tensor \p Tl and \p Tr.
    Furthermore, the function assume two tensor has the same rank. In case where two tensors have
    different ranks, the small one will be extend by adding redundant dimension to the beginning of
    axis (T<x>_pad_right=true) or by adding redundant dim to the last axis (if T<x>_pad_left=false
    [default]). if the Tensor #1 has shape=(i1,j1,k1,l1...), and Tensor #2 has
    shape=(i2,j2,k2,l2...); then the return Tensor will have shape=(i1*i2,j1*j2,k1*k2...)
    @param[in] Tl rank-n Tensor #1
    @param[in] Tr rank-m Tensor #2
    @param[in] Tl_pad_left The padding scheme for Tl if Tl.rank != Tr.rank
    @param[in] Tr_pad_left The padding scheme for Tr if Tl.rank != Tr.rank
    @return [Tensor]
    @pre two tensor should on same device.

    */
    Tensor Kron(const Tensor &Tl, const Tensor &Tr, const bool &Tl_pad_left = false,
                const bool &Tr_pad_left = false);

    // Directsum:
    //==================================================
    /**
    @brief perform directsum of two Tensor.
    @details
        The function assume two tensor has the same rank, and axes indicated in <shared_axes> are
    the same for both T1 and T2. The out put tensors will have same rank as T1 and T2, with the
    dimension of rest of the axes being the sum of dimensions of T1 and T2. e.g., the out put shape
    = (i1+i2,j1+j2, share_axis_1, k1+k2, share_axis_2, ...); where T1.shape =
    (i1,j1,share_axis_1,k1,share_axis_2 ...) and T2.shape = (i2,j2,share_axis_1,k2,share_axis_2 ...)
    @param[in] T1 rank-n Tensor #1
    @param[in] T2 rank-n Tensor #2
    @param[in] shared_axes The axes that are shared by two tensors
    @return [Tensor]
    @pre two tensor should on same device.
    */
    Tensor Directsum(const Tensor &T1, const Tensor &T2,
                     const std::vector<cytnx_uint64> &shared_axes);

    // VectorDot:
    //=================================================
    /**
    @brief perform inner product of vectors
    @param[in] Tl Tensor #1
    @param[in] Tr Tensor #2
    @param[in] if the Tl should be conjugated (only work for complex. For real Tensor, no function),
    default: false
    @return [Tensor] Rank-0
    @pre two Tensors must be Rank-1, with same length.
    @note performance tune: This function have better performance when two vectors with same types,
    and are one of following type: cytnx_double, cytnx_float, cytnx_complex64 or cytnx_complex128.
    */
    Tensor Vectordot(const Tensor &Tl, const Tensor &Tr, const bool &is_conj = false);

    // Dot:
    //=================================================
    /**
    @brief dot product of two arrays.
    @details
        1. if both Tl and Tr are 1d arrays, it is inner product of vectors (no complex conj), it
    calls linalg.Vectordot with is_conj=false.
        2. if both Tl and Tr are 2d arrays, it calls linalg.Matmul to compute the matrix
    multiplication
        3. if Tl is Nd array (with N>=2, and Tr is 1-D array, it is sum product over the last axis
    of a with b
    @param[in] Tl Tensor #1
    @param[in] Tr Tensor #2
    @return
        [Tensor]
    @note
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
    @param[in] Diag Tensor #1
    @param[in] Sub_diag Tensor #2
    @param[in] is_V: if calculate the eigen value.
    @param[in] k: Return k lowest eigen vector if is_V=True
    @param[in] throw_excp: Whether to throw exception when error occurs in Tridiag internal function
    @return
        [vector<Tensor>] if is_V = True, the first tensor is the eigen value, and second tensor is
    eigenvector of shape [k,L].

    @pre
        two Tensors must be Rank-1, with length of Diag = L and Sub_diag length = L-1.

    @note
        performance tune: This function have better performance when two vectors with same types,
    and are one of following type: cytnx_double, cytnx_float. In general all real type can be use as
    input, which will be promote to floating point type for calculation.

    */
    std::vector<Tensor> Tridiag(const Tensor &Diag, const Tensor &Sub_diag, const bool &is_V = true,
                                const bool &is_row = false, bool throw_excp = false);

    // ExpH:
    //===========================================
    /**
    @brief perform matrix exponential for Hermitian matrix
    @details This function perform matrix exponential for Hermitian matrix, That is,
    \f[
        O = \exp{(aM + b)}
    \f]
    @param[in] in input Tensor, should be Hermitian
    @param[in] a rescale factor
    @param[in] b bias
    @return
        [Tensor]
    @pre the \p in should be a Hermitian matrix.
    @warning If \p in is not a Hermitian matrix, only the lower triangular matrix will be used.
    (This is strongly not recommended, please use ExpM(const Tensor &in, const T &a, const T &b)
    instead).

    */
    template <typename T>
    Tensor ExpH(const Tensor &in, const T &a, const T &b = 0);
    /**
     * @brief perform matrix exponential for Hermitian matrix
     * @details This function perform matrix exponential for Hermitian matrix, That is,
     * \f[
     *    O = \exp{M}
     * \f]
     *@pre the \p in should be a Hermitian matrix.
     *@warning If \p in is not a Hermitian matrix, only the lower triangular matrix will be used.
    (This is strongly not recommended, please use ExpM(const Tensor &in) instead).

     * @see ExpH(const Tensor &in, const T &a, const T &b = 0)
    */
    Tensor ExpH(const Tensor &in);

    // ExpM:
    //===========================================
    /**
    @brief perform matrix exponential for generic matrix
    @details This function perform matrix exponential for generic matrix, That is,
    \f[
        O = \exp{(aM + b)}
    \f]
    @param[in] in input Tensor, should be a square rank-2.
    @param[in] a rescale factor
    @param[in] b bias
    @pre the \p in should be a square matrix.
    @return [Tensor]
    */
    template <typename T>
    Tensor ExpM(const Tensor &in, const T &a, const T &b = 0);

    /**
     * @brief perform matrix exponential for generic matrix
     * @details This function perform matrix exponential for generic matrix, That is,
     * \f[
     *     O = \exp{M}
     * \f]
     * @param[in] in input Tensor, should be a square rank-2.
     * @return [Tensor]
     * @see ExpM(const Tensor &in, const T &a, const T &b = 0)
     */
    Tensor ExpM(const Tensor &in);

    // Arnoldi:
    //===========================================
    /**
    @brief perform Arnoldi for matrices or linear function.
    @details This function calculate the eigen value problem using Arnoldi algorithm.
    @param[in] Hop the Linear Operator defined by LinOp class or it's inheritance (see LinOp).
    @param[in] Tin the initial vector, this should be rank-1.
    @param[in] which
    @parblock
    which order eigenvlues and corresponding eigenvectors should be find, the supported
    options are:

    <b>'LM'</b> : largest magnitude
    <b>'LR'</b> : largest real part
    <b>'LI'</b> : largest imaginary part
    <b>'SM'</b> : smallest magnitude
    <b>'SR'</b> : smallest real part
    <b>'SI'</b> : smallest imaginary part

    @endparblock
    @param[in] maxiter the maximum interation steps for each k.
    @param[in] cvg_crit the convergence criterion of the energy.
    @param[in] k the number of lowest k eigen values.
    @param[in] is_V if set to true, the eigen vectors will be returned.
    @param[in] verbose print out iteration info.
    @return
        [eigvals (Tensor), eigvecs (Tensor)(option)]
    @note
        To use, define a linear operator with LinOp class either by assign a custom function or
    create a class that inherit LinOp (see LinOp for further details)
    */
    std::vector<Tensor> Arnoldi(LinOp *Hop, const Tensor &Tin = Tensor(),
                                const std::string which = "LM", const cytnx_uint64 &maxiter = 10000,
                                const cytnx_double &cvg_crit = 1.0e-9, const cytnx_uint64 &k = 1,
                                const bool &is_V = true, const bool &verbose = false);

    // Arnoldi:
    //===========================================
    /**
    @brief perform Arnoldi for matrices or linear function.
    @details This function calculate the eigen value problem using Arnoldi algorithm.
    @param[in] Hop the Linear Operator defined by LinOp class or it's inheritance (see LinOp).
    @param[in] Tin the initial UniTensor.
    @param[in] which
    @parblock
    which order eigenvlues and corresponding eigenvectors should be find, the supported
    options are:

    <b>'LM'</b> : largest magnitude
    <b>'LR'</b> : largest real part
    <b>'LI'</b> : largest imaginary part
    <b>'SR'</b> : smallest real part
    <b>'SI'</b> : smallest imaginary part

    @endparblock
    @param[in] maxiter the maximum interation steps for each k.
    @param[in] cvg_crit the convergence criterion of the energy.
    @param[in] k the number of lowest k eigen values.
    @param[in] is_V if set to true, the eigen vectors will be returned.
    @param[in] verbose print out iteration info.
    @return
        [eigvals (UniTensor), eigvec_1, eivec_2, ..., eigvec_k].
                The first UniTensor contains eigenvalues.
    @note
        To use, define a linear operator with LinOp class either by assign a custom function or
    create a class that inherit LinOp (see LinOp for further details)

        @pre
        1. The initial UniTensor cannot be empty.
        2. The UniTensor version of the Arnoldi not support \p which = 'SM'.
    */
    std::vector<UniTensor> Arnoldi(LinOp *Hop, const UniTensor &Tin, const std::string which = "LM",
                                   const cytnx_uint64 &maxiter = 10000,
                                   const cytnx_double &cvg_crit = 1.0e-9, const cytnx_uint64 &k = 1,
                                   const bool &is_V = true, const bool &verbose = false);

    // Lanczos:
    //===========================================
    /**
    @brief perform Lanczos for hermitian/symmetric matrices or linear function.
    @details
        This function calculate the eigen value problem using explicitly restarted Lanczos.
    #Performance tune:
        For small linear dimension, try to reduce max_krydim.
    @param[in] Hop the Linear Operator defined by LinOp class or it's inheritance (see LinOp).
    @param[in] Tin the initial vector, this should be rank-1.
    @param[in] method
    @parblock
    the desired Lanczos method to use, the supported options are:

    <b>'ER'</b> : explicitly restarted Lanczos
    <b>'Gnd'</b> : naive Lanczos

    @endparblock

    @param[in] CvgCrit the convergence criterion of the energy.
    @param[in] maxiter the maximum interation steps for each k.
    @param[in] k the number of lowest k eigen values.
    @param[in] is_V if set to true, the eigen vectors will be returned.
    @param[in] is_row whether the return eigen vectors should be in row-major form.
    @param[in] max_krydim the maximum krylov subspace dimension for each iteration.
    @param[in] verbose print out iteration info.
    @return
        [eigvals (Tensor), eigvecs (Tensor)(option)]
    @note
    To use, define a linear operator with LinOp class either by assign a custom function or
    create a class that inherit LinOp (see LinOp for further details)
    */
    std::vector<Tensor> Lanczos(LinOp *Hop, const Tensor &Tin = Tensor(),
                                const std::string method = "Gnd", const double &CvgCrit = 1.0e-14,
                                const unsigned int &Maxiter = 10000, const cytnx_uint64 &k = 1,
                                const bool &is_V = true, const bool &is_row = false,
                                const cytnx_uint32 &max_krydim = 0, const bool &verbose = false);

    // Lanczos:
    //===========================================
    /**
    @brief perform Lanczos for hermitian/symmetric matrices or linear function.
    @details
        This function calculate the eigen value problem using explicitly restarted Lanczos.
    #Performance tune:
        For small linear dimension, try to reduce max_krydim.
    @param[in] Hop the Linear Operator defined by LinOp class or it's inheritance (see LinOp).
    @param[in] Tin the initial vector, this should be rank-1.
    @param[in] method
    @parblock
    the desired Lanczos method to use, the supported options are:

    <b>'ER'</b> : explicitly restarted Lanczos
    <b>'Gnd'</b> : naive Lanczos

    @endparblock

    @param[in] CvgCrit the convergence criterion of the energy.
    @param[in] maxiter the maximum interation steps for each k.
    @param[in] k the number of lowest k eigen values.
    @param[in] is_V if set to true, the eigen vectors will be returned.
    @param[in] is_row whether the return eigen vectors should be in row-major form.
    @param[in] max_krydim the maximum krylov subspace dimension for each iteration.
    @param[in] verbose print out iteration info.
    @return
        [eigvals (UniTensor), eigvecs (UniTensor)(option)]
    @note
    To use, define a linear operator with LinOp class either by assign a custom function or
    create a class that inherit LinOp (see LinOp for further details)
    */
    std::vector<UniTensor> Lanczos(LinOp *Hop, const UniTensor &Tin = UniTensor(),
                                   const std::string method = "Gnd",
                                   const double &CvgCrit = 1.0e-14,
                                   const unsigned int &Maxiter = 10000, const cytnx_uint64 &k = 1,
                                   const bool &is_V = true, const bool &is_row = false,
                                   const cytnx_uint32 &max_krydim = 4, const bool &verbose = false);

    // Lanczos:
    //===========================================
    /**
    @brief perform Lanczos for hermitian/symmetric matrices or linear function.
    @details
        This function calculate the eigen value problem using explicitly restarted Lanczos.
    #Performance tune:
        For small linear dimension, try to reduce max_krydim.
    @param[in] Hop the Linear Operator defined by LinOp class or it's inheritance (see LinOp).
    @param[in] k the number of lowest k eigen values.
    @param[in] is_V if set to true, the eigen vectors will be returned.
    @param[in] maxiter the maximum interation steps for each k.
    @param[in] CvgCrit the convergence criterion of the energy.
    @param[in] is_row whether the return eigen vectors should be in row-major form.
    @param[in] Tin the initial vector, this should be rank-1
    @param[in] max_krydim the maximum krylov subspace dimension for each iteration.
    @param[in] verbose print out iteration info.
    @return
        [eigvals (Tensor), eigvecs (Tensor)(option)]
    @note
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
    @details
        This function calculate the eigen value problem using naive Lanczos to get ground state and
    lowest eigen value.
    @param[in] Hop the Linear Operator defined by LinOp class or it's inheritance (see LinOp).
    @param[in] CvgCrit the convergence criterion of the energy.
    @param[in] is_V if set to true, the eigen vectors will be returned.
    @param[in] Tin the initial vector, this should be rank-1
    @param[in] verbose print out iteration info.
    @param[in] maxiter the maximum interation steps for each k.
    @return
        [eigvals (Tensor), eigvecs (Tensor)(option)]
    @note
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
    @details
        This function calculate the eigen value problem using naive Lanczos to get ground state and
    lowest eigen value.
    @param[in] Hop the Linear Operator defined by LinOp class or it's inheritance (see LinOp).
    @param[in] CvgCrit the convergence criterion of the energy.
    @param[in] is_V if set to true, the eigen vectors will be returned.
    @param[in] Tin the initial vector, this should be a UniTensor.
    @param[in] verbose print out iteration info.
    @param[in] maxiter the maximum interation steps for each k.
    @return
        [eigvals (UniTensor::Dense), eigvecs (UniTensor)(option)]
    @note
        To use, define a linear operator with LinOp class either by assign a custom function or
    create a class that inherit LinOp (see LinOp for further details)
    */
    std::vector<UniTensor> Lanczos_Gnd_Ut(LinOp *Hop, const UniTensor &Tin,
                                          const double &CvgCrit = 1.0e-14, const bool &is_V = true,
                                          const bool &verbose = false,
                                          const unsigned int &Maxiter = 100000);

    // Lanczos_Exp:
    //===============================================
    /**
    @brief Perform the Lanczos algorithm for hermitian operator
    \f$H\f$ to approximate \f$e^{H\tau}v\f$.
    @details
        This function perform the Lanczos-like algorithm for hermitian
                linear operator \f$H\f$ to approximate
                \f[
                e^{H\tau}v
                \f] and return the state \f$w\f$ such that
                \f[
                |\exp(H\tau)v - w| < \delta.
                \f]
                Here \f$v\f$ is a given vector or a state.
    @param[in] Hop the Linear Operator defined by LinOp class or it's inheritance (see LinOp). The
    operation method \f$Hv\f$ need to be defined in it.
    @param[in] v The input vector (or state). The norm \f$|v|\f$ should be equal to 1.
    @param[in] tau A scalar, it can be complex number.
    @param[in] CvgCrit \f$\delta\f$, the convergence criterion.
    @param[in] Maxiter the maximum interation steps for each k.
    @param[in] verbose print out iteration info.
    @return
        UniTensor \f$w\f$
    @note
        To use, define a linear operator with LinOp class either by assign a custom function or
    create a class that inherit LinOp (see LinOp for further details)
        @warning
                User need to guarantee that the input operator \f$H\f$ is Hermitian
    , and the exponetiate \f$e^{-H\tau}\f$ will converged. Ohterwise, the function will return the
    wrong results without any warning.
    */
    UniTensor Lanczos_Exp(LinOp *Hop, const UniTensor &v, const Scalar &tau,
                          const double &CvgCrit = 1.0e-10, const unsigned int &Maxiter = 100000,
                          const bool &verbose = false);

    // Lstsq:
    //===========================================
    /**
    @brief Return the least-squares solution to a linear matrix equation.
    @details
        Computes the vector x that approximatively solves the equation A @ x = b. The equation may
    be under-, well-, or over-determined independent columns. If a is square and of full rank, then
    x (but for round-off error) is the “exact” solution of the equation. Else, x minimizes the
    Euclidean 2-norm \f$ || b - a x ||_2 \f$.

    @param[in] A “Coefficient” matrix, must be two-dimensional.
    @param[in] b Ordinate or “dependent variable” values, must be two-dimensional, the least-squares
    solution is calculated for each of the K columns of b.
    @param[in] rcond Cut-off ratio for small singular values of a. For the purposes of rank
    determination, singular values are treated as zero if they are smaller than rcond times the
    largest singular value of A, If it is negative, the machine precision is used.
    @return
    [std::vector<Tensors>]

    1. the first tensor is least-squares solutions in the K columns.
    2. the second tensor is the sums of squared residuals: Squared Euclidean 2-norm for each
    column in b - a @ x. If the rank of a is < N or M <= N, this is a zero Tensor.
    3. the third tensor is the rank of matrix A.
    4. the forth tensor is singular values of A.

    @author Ke
    */
    std::vector<Tensor> Lstsq(const Tensor &A, const Tensor &b, const float &rcond = -1);

    /**
    @brief Blas Axpy, performing \f$ a\textbf{x} + \textbf{y} \f$, inplacely.
    @details
    This function performs
    \f[
    a\textbf{x} + \textbf{y},
    \f]
    where \f$ \textbf{x},\textbf{y} \f$ are Tensor and \f$ a \f$ is a Scalar. The dtype of return
    Tensor will be the strongest among \p x, \p y and \p a.
    @param[in] a Scalar.
    @param[in] x Tensor, can be any rank
    @param[in] y Tensor, can be any rank
    @return
    [Tensor]
    If \f$ \textbf{y} \f$ is not specify, then it performs \f$ a\textbf{x} \f$ -> return
    @note This will return a new tensor.
    */
    Tensor Axpy(const Scalar &a, const Tensor &x, const Tensor &y = Tensor());

    /**
     * @brief Blas Axpy, performing \f$ \textbf{y} = a\textbf{x} + \textbf{y} \f$, inplacely.
     * @details
     * This function performs
     * \f[
     * \textbf{y} = a\textbf{x} + \textbf{y},
     * \f]
     * where \f$ \textbf{x},\textbf{y} \f$ are Tensor and a is a Scalar. The dtype of return
     * Tensor will be the strongest among \p x, \p y and \p a.
     * @param[in ] a Scalar.
     * @param[in ] x Tensor, can be any rank
     * @param[in ] y Tensor, can be any rank
     * @return
     * [Tensor]
     * If \f$ \textbf{y} \f$ is not specify, then it performs \f$ a\textbf{x} \f$ -> return
     * @note Compared to Axpy(const Scalar &a, const Tensor &x, const Tensor &y = Tensor()), this
     * function will perform inplacely.
     */
    void Axpy_(const Scalar &a, const Tensor &x, Tensor &y);

    /**
    @brief Blas Ger, performing return = a*vec(x)*vec(y)^T
    @details
        This function performs a*x*y^T where x,y are rank-1 Tensor with dimension nx and ny
    respectively; and a is a Scalar. The dtype of return Tensor will be the strongest among x,y and
    a.
    @param[in] x Tensor, rank-1 with size nx
    @param[in] y Tensor, rank-1 with size ny
    @param[in] a Scalar, if not provided a = 1.
    @return
        [Tensor with shape (nx,ny)]
    @note
        This will return a new tensor.
    */
    Tensor Ger(const Tensor &x, const Tensor &y, const Scalar &a = Scalar());

    /**
     * @brief Blas Gemm, performing \f$ \textbf{c} = a\textbf{x}\textbf{y} + b\textbf{c} \f$,
     * inplacely.
     * @details
     * This function performs
     * \f[
     * \textbf{c} = a\textbf{x}\textbf{y} + b\textbf{c},
     * \f]
     * where \f$ \textbf{x},\textbf{y},\textbf{c} \f$ are rank-2 Tensor and a, b are Scalars. The
     * dtype of return Tensor will be the strongest among \p x, \p y and \p a \p b \p c.
     * @param[in ] a Scalar.
     * @param[in ] x Tensor, rank-2 with shape (M,N)
     * @param[in ] y Tensor, rank-2 with shape (N,K)
     * @param[in ] b Scalar.
     * @param[in,out ] c Tensor, rank-2 with shape (M,K), must be properly initialized with the
     * correct shape.
     */
    void Gemm_(const Scalar &a, const Tensor &x, const Tensor &y, const Scalar &b, Tensor &c);

    /**
     * @brief Blas Gemm, performing \f$ a\textbf{x}\textbf{y} -> \f$ return.
     * @details
     * This function performs
     * \f[
     * return = a\textbf{x}\textbf{y},
     * \f]
     * where \f$ \textbf{x},\textbf{y} \f$ are rank-2 Tensor and a is Scalar. The dtype of return
     * Tensor will be the strongest among \p x, \p y and \p a.
     * @param[in ] a Scalar.
     * @param[in ] x Tensor, rank-2 with shape (M,N)
     * @param[in ] y Tensor, rank-2 with shape (N,K)
     * @return
     * [Tensor] with shape (M,K)
     */
    Tensor Gemm(const Scalar &a, const Tensor &x, const Tensor &y);

    /**
     * @brief Blas Gemm_Batch, performing many(batch) \f$ \textbf{c} = \alpha\textbf{a}\textbf{b} +
     * \beta\textbf{c} \f$, inplacely. You do not need to consider the row-major or column-major,
     * just provide the correct shape.
     * @details
     * see
     * https://www.intel.com/content/www/us/en/developer/articles/technical/introducing-batch-gemm-operations.html
     * This function performs tensor type check and type conversion, then call the corresponding
     * blas function.
     *
     * @param[in] m_array array of cytnx_int64, each element is the number of rows of \p a_tensors
     * @param[in] n_array array of cytnx_int64, each element is the number of columns of \p
     * b_tensors
     * @param[in] k_array array of cytnx_int64, each element is the number of columns of \p
     * a_tensors and the number of rows of \p b_tensors
     * @param[in] alpha_array array of Scalar, each element is the scalar \p alpha
     * @param[in] a_tensors array of Tensor, each element is a rank-2 Tensor with shape
     * (m_array[i],k_array[i])
     * @param[in] b_tensors array of Tensor, each element is a rank-2 Tensor with shape
     * (k_array[i],n_array[i])
     * @param[in] beta_array array of Scalar, each element is the scalar \p beta
     * @param[in,out] c_tensors array of Tensor, each element is a rank-2 Tensor with shape
     * (m_array[i],n_array[i]), \b{must be properly initialized with the correct shape}.
     * @param[in] group_count cytnx_int64, the number of groups
     * @param[in] group_size array of cytnx_int64, each element is the number of matrices in each
     * group
     */
    void Gemm_Batch(const std::vector<cytnx_int64> &m_array,
                    const std::vector<cytnx_int64> &n_array,
                    const std::vector<cytnx_int64> &k_array, const std::vector<Scalar> &alpha_array,
                    const std::vector<Tensor> &a_tensors, const std::vector<Tensor> &b_tensors,
                    const std::vector<Scalar> &beta_array, std::vector<Tensor> &c_tensors,
                    const cytnx_int64 group_count, const std::vector<cytnx_int64> &group_size);

    ///@cond
    void __Gemm_Batch(const std::vector<char> &transa_array, const std::vector<char> &transb_array,
                      const std::vector<blas_int> &m_array, const std::vector<blas_int> &n_array,
                      const std::vector<blas_int> &k_array, const std::vector<Scalar> &alpha_array,
                      const void **a_array, const void **b_array,
                      const std::vector<Scalar> &beta_array, void **c_array,
                      const blas_int group_count, const std::vector<blas_int> &group_size,
                      const unsigned int dtype, const int device);

    ///@endcond

  }  // namespace linalg

  // operators:
  /**
   * @brief The addition operator for Tensor.
   * @details This operator call linalg::Add(const Tensor &Lt, const Tensor &Rt)
   * to perform the addition.
   * @param[in] Lt Left Tensor.
   * @param[in] Rt Right Tensor.
   * @return [Tensor] the result of addition.
   * @pre \p Lt and \p Rt must have the same shape.
   * @see linalg::Add(const Tensor &Lt, const Tensor &Rt)
   */
  Tensor operator+(const Tensor &Lt, const Tensor &Rt);

  /**
   * @brief The addition operator between a template type and Tensor.
   * @details This operator call linalg::Add(const T &lc, const Tensor &Rt) to
   * perform the addition between a template type and Tensor.
   * @param[in] lc Left template type.
   * @param[in] Rt Right Tensor.
   * @return [Tensor] the result of addition.
   * @see linalg::Add(const T &lc, const Tensor &Rt)
   */
  template <class T>
  Tensor operator+(const T &lc, const Tensor &Rt);

  /**
   * @brief The addition operator between Tensor and a template type.
   * @details This operator call linalg::Add(const Tensor &Lt, const T &rc) to
   * perform the addition between Tensor and a template type.
   * @param[in] Lt Left Tensor.
   * @param[in] rc Right template type.
   * @return [Tensor] the result of addition.
   * @see linalg::Add(const Tensor &Lt, const T &rc)
   */
  template <class T>
  Tensor operator+(const Tensor &Lt, const T &rc);

  //------------------------------------
  /**
   * @brief The subtraction operator for Tensor.
   * @details This operator call linalg::Sub(const Tensor &Lt, const Tensor &Rt)
   * to perform the subtraction.
   * @param[in] Lt Left Tensor.
   * @param[in] Rt Right Tensor.
   * @return [Tensor] the result of subtraction.
   * @pre \p Lt and \p Rt must have the same shape.
   * @see linalg::Sub(const Tensor &Lt, const Tensor &Rt)
   */
  Tensor operator-(const Tensor &Lt, const Tensor &Rt);

  /**
   * @brief The subtraction operator between a template type and Tensor.
   * @details This operator call linalg::Sub(const T &lc, const Tensor &Rt) to
   * perform the subtraction between a template type and Tensor.
   * @param[in] lc Left template type.
   * @param[in] Rt Right Tensor.
   * @return [Tensor] the result of subtraction.
   * @see linalg::Sub(const T &lc, const Tensor &Rt)
   */
  template <class T>
  Tensor operator-(const T &lc, const Tensor &Rt);

  /**
   * @brief The subtraction operator between Tensor and a template type.
   * @details This operator call linalg::Sub(const Tensor &Lt, const T &rc) to
   * perform the subtraction between Tensor and a template type.
   * @param[in] Lt Left Tensor.
   * @param[in] rc Right template type.
   * @return [Tensor] the result of subtraction.
   * @see linalg::Sub(const Tensor &Lt, const T &rc)
   */
  template <class T>
  Tensor operator-(const Tensor &Lt, const T &rc);

  //-----------------------------------
  /**
   * @brief The multiplication operator for Tensor.
   * @details This operator call linalg::Mul(const Tensor &Lt, const Tensor &Rt)
   * to perform the multiplication.
   * @param[in] Lt Left Tensor.
   * @param[in] Rt Right Tensor.
   * @return [Tensor] the result of multiplication.
   * @pre \p Lt and \p Rt must have the same shape.
   * @see linalg::Mul(const Tensor &Lt, const Tensor &Rt)
   */
  Tensor operator*(const Tensor &Lt, const Tensor &Rt);

  /**
   * @brief The multiplication operator between a template type and Tensor.
   * @details This operator call linalg::Mul(const T &lc, const Tensor &Rt) to
   * perform the multiplication between a template type and Tensor.
   * @param[in] lc Left template type.
   * @param[in] Rt Right Tensor.
   * @return [Tensor] the result of multiplication.
   * @see linalg::Mul(const T &lc, const Tensor &Rt)
   */
  template <class T>
  Tensor operator*(const T &lc, const Tensor &Rt);

  /**
   * @brief The multiplication operator between Tensor and a template type.
   * @details This operator call linalg::Mul(const Tensor &Lt, const T &rc) to
   * perform the multiplication between Tensor and a template type.
   * @param[in] Lt Left Tensor.
   * @param[in] rc Right template type.
   * @return [Tensor] the result of multiplication.
   * @see linalg::Mul(const Tensor &Lt, const T &rc)
   */
  template <class T>
  Tensor operator*(const Tensor &Lt, const T &rc);

  //----------------------------------
  /**
   * @brief The division operator for Tensor.
   * @details This operator call linalg::Div(const Tensor &Lt, const Tensor &Rt)
   * to perform the division.
   * @param[in] Lt Left Tensor.
   * @param[in] Rt Right Tensor.
   * @return [Tensor] the result of division.
   * @see linalg::Div(const Tensor &Lt, const Tensor &Rt)
   * @pre
   * 1. The divisor cannot be zero.
   * 2. \p Lt and \p Rt must have the same shape.
   */
  Tensor operator/(const Tensor &Lt, const Tensor &Rt);

  /**
   * @brief The division operator between a template type and Tensor.
   * @details This operator call linalg::Div(const T &lc, const Tensor &Rt) to
   * perform the division between a template type and Tensor.
   * @param[in] lc Left template type.
   * @param[in] Rt Right Tensor.
   * @return [Tensor] the result of division.
   * @see linalg::Div(const T &lc, const Tensor &Rt)
   * @pre The divisor cannot be zero.
   */
  template <class T>
  Tensor operator/(const T &lc, const Tensor &Rt);

  /**
   * @brief The division operator between Tensor and a template type.
   * @details This operator call linalg::Div(const Tensor &Lt, const T &rc) to
   * perform the division between Tensor and a template type.
   * @param[in] Lt Left Tensor.
   * @param[in] rc Right template type.
   * @return [Tensor] the result of division.
   * @see linalg::Div(const Tensor &Lt, const T &rc)
   * @pre The divisor cannot be zero.
   */
  template <class T>
  Tensor operator/(const Tensor &Lt, const T &rc);

  //----------------------------------
  /**
   * @brief The mode operator for Tensor.
   * @details This operator call linalg::Mod(const Tensor &Lt, const Tensor &Rt)
   * to perform the mode.
   * @param[in] Lt Left Tensor.
   * @param[in] Rt Right Tensor.
   * @return [Tensor] the result of mode.
   * @pre \p Lt and \p Rt must have the same shape.
   * @see linalg::Mod(const Tensor &Lt, const Tensor &Rt)
   */
  Tensor operator%(const Tensor &Lt, const Tensor &Rt);

  /**
   * @brief The mode operator between a template type and Tensor.
   * @details This operator call linalg::Mod(const T &lc, const Tensor &Rt) to
   * perform the mode between a template type and Tensor.
   * @param[in] lc Left template type.
   * @param[in] Rt Right Tensor.
   * @return [Tensor] the result of mode.
   * @see linalg::Mod(const T &lc, const Tensor &Rt)
   */
  template <class T>
  Tensor operator%(const T &lc, const Tensor &Rt);

  /**
   * @brief The mode operator between Tensor and a template type.
   * @details This operator call linalg::Mod(const Tensor &Lt, const T &rc) to
   * perform the mode between Tensor and a template type.
   * @param[in] Lt Left Tensor.
   * @param[in] rc Right template type.
   * @return [Tensor] the result of mode.
   * @see linalg::Mod(const Tensor &Lt, const T &rc)
   */
  template <class T>
  Tensor operator%(const Tensor &Lt, const T &rc);

  //----------------------------------
  /**
   * @brief The comparison operator for Tensor.
   * @details This operator call linalg::Cpr(const Tensor &Lt, const Tensor &Rt)
   * to perform the comparison.
   * @param[in] Lt Left Tensor.
   * @param[in] Rt Right Tensor.
   * @return [Tensor] the result of comparison.
   * @see linalg::Cpr(const Tensor &Lt, const Tensor &Rt)
   */
  Tensor operator==(const Tensor &Lt, const Tensor &Rt);

  /**
   * @brief The comparison operator between a template type and Tensor.
   * @details This operator call linalg::Cpr(const T &lc, const Tensor &Rt) to
   * perform the comparison between a template type and Tensor.
   * @param[in] lc Left template type.
   * @param[in] Rt Right Tensor.
   * @return [Tensor] the result of comparison.
   * @see linalg::Cpr(const T &lc, const Tensor &Rt)
   */
  template <class T>
  Tensor operator==(const T &lc, const Tensor &Rt);

  /**
   * @brief The comparison operator between Tensor and a template type.
   * @details This operator call linalg::Cpr(const Tensor &Lt, const T &rc) to
   * perform the comparison between Tensor and a template type.
   * @param[in] Lt Left Tensor.
   * @param[in] rc Right template type.
   * @return [Tensor] the result of comparison.
   * @see linalg::Cpr(const Tensor &Lt, const T &rc)
   */
  template <class T>
  Tensor operator==(const Tensor &Lt, const T &rc);

}  // namespace cytnx

#endif  // BACKEND_TORCH

#endif
