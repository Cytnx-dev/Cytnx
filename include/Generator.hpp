#ifndef CYTNX_GENERATOR_H_
#define CYTNX_GENERATOR_H_

#include "Type.hpp"
#include "Device.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include <vector>
#include <initializer_list>

#ifdef BACKEND_TORCH
#else
namespace cytnx {

  //@{
  /**
  @brief Create a Tensor with all elements initialized to zero.
  @param shape the shape of the Tensor
  @param dtype the dtype of the Tensor. It can be any type defined in \link cytnx::Type cytnx::Type
  \endlink
  @param device the device that the Tensor is put on. It can be any device defined in \link
  cytnx::Device cytnx::Device \endlink

  @return
      [Tensor]

  */
  Tensor zeros(const std::vector<cytnx_uint64> &shape, unsigned int dtype = Type.Double,
               int device = Device.cpu);
  Tensor zeros(std::initializer_list<cytnx_uint64> shape, unsigned int dtype = Type.Double,
               int device = Device.cpu);
  //@}

  /**
  @brief Create a square rank-2 Tensor with the diagonal initialized to one and all other elements
  set to zero.
  @param Dim the dimension of diagonal.
  @param dtype the dtype of the Tensor. It can be any type defined in \link cytnx::Type cytnx::Type
  \endlink
  @param device the device that the Tensor is put on. It can be any device defined in \link
  cytnx::Device cytnx::Device \endlink

  @return
      [Tensor]

  */
  Tensor identity(cytnx_uint64 Dim, unsigned int dtype = Type.Double, int device = Device.cpu);

  /**
  @brief Create a square rank-2 Tensor with the diagonal initialized to one and all other elements
  set to zero
  @param Dim the dimension of diagonal.
  @param dtype the dtype of the Tensor. It can be any type defined in \link cytnx::Type cytnx::Type
  \endlink
  @param device the device that the Tensor is put on. It can be any device defined in \link
  cytnx::Device cytnx::Device \endlink

  @return
      [Tensor]

  Note:
      This function is a alias of cytnx::identity().

  */
  Tensor eye(cytnx_uint64 Dim, unsigned int dtype = Type.Double, int device = Device.cpu);

  //@{
  /**
  @brief Create a Tensor with all elements initialized to one.
  @param shape the shape of the Tensor
  @param dtype the dtype of the Tensor. It can be any type defined in \link cytnx::Type cytnx::Type
  \endlink
  @param device the device that the Tensor is put on. It can be any device defined in \link
  cytnx::Device cytnx::Device \endlink

  @return
      [Tensor]

  */
  Tensor ones(const std::vector<cytnx_uint64> &shape, unsigned int dtype = Type.Double,
              int device = Device.cpu);
  Tensor ones(std::initializer_list<cytnx_uint64> shape, unsigned int dtype = Type.Double,
              int device = Device.cpu);
  //@}

  //@{
  /**
  @brief Create a rank-1 Tensor with incremental unsigned integer elements in the range [0,Nelem).
  @details The elements are 0, 1, 2, ..., \p Nelem - 1.
  @param[in] Nelem number of incremental elements.
  @return
      [Tensor]
  */
  Tensor arange(cytnx_int64 Nelem);
  /**
  @brief Create a rank-1 Tensor with incremental elements \p start, \p start + \p step,
  \p start + 2 * \p step, ..., following the same convention as
  [numpy.arange](https://numpy.org/doc/stable/reference/generated/numpy.arange.html).
  @details The number of elements is `ceil((end - start) / step)`. The range is nominally
  half-open [\p start, \p end), but -- exactly as with numpy -- this is not guaranteed under
  floating-point rounding: the final element may equal or slightly exceed \p end (e.g.
  `arange(0.5, 0.8, 0.1)` yields `[0.5, 0.6, 0.7, 0.8]`). An empty or direction-mismatched range
  (a non-positive count) yields a zero-extent Tensor.
  @param[in] start start value of the range.
  @param[in] end end value of the range (nominally exclusive; see details).
  @param[in] step step-size between subsequent elements; must be non-zero.
  @param[in] dtype the data type of the Tensor, see cytnx::Type for more information.
  @param[in] device the device type of the Tensor, see cytnx::Device for more information.
  @return
      [Tensor]
  */
  Tensor arange(cytnx_double start, cytnx_double end, cytnx_double step = 1,
                unsigned int dtype = Type.Double, int device = Device.cpu);
  //@}

  Tensor linspace(cytnx_double start, cytnx_double end, cytnx_uint64 Nelem, bool endpoint = true,
                  unsigned int dtype = Type.Double, int device = Device.cpu);

  // Tensor rand(const std::vector<cytnx_uint64> &size);
  // Tensor randn(const std::vector<cytnx_uint64> &size);
  // Tensor normal(const cytnx_double &mean, cytnx_double &std);

}  // namespace cytnx

#endif  // BACKEND_TORCH

#endif  // CYTNX_GENERATOR_H_
