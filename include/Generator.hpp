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
  Tensor zeros(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype = Type.Double,
               const int &device = Device.cpu);
  Tensor zeros(std::initializer_list<cytnx_uint64> shape, const unsigned int &dtype = Type.Double,
               const int &device = Device.cpu);
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
  Tensor identity(const cytnx_uint64 &Dim, const unsigned int &dtype = Type.Double,
                  const int &device = Device.cpu);

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
  Tensor eye(const cytnx_uint64 &Dim, const unsigned int &dtype = Type.Double,
             const int &device = Device.cpu);

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
  Tensor ones(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype = Type.Double,
              const int &device = Device.cpu);
  Tensor ones(std::initializer_list<cytnx_uint64> shape, const unsigned int &dtype = Type.Double,
              const int &device = Device.cpu);
  //@}

  //@{
  /**
  @brief Create a rank-1 Tensor with incremental unsigned integer elements in the range [0,Nelem).
  @details The elements are 0, 1, 2, ..., \p Nelem - 1.
  @param[in] Nelem number of incremental elements.
  @return
      [Tensor]
  */
  Tensor arange(const cytnx_int64 &Nelem);
  /**
  @brief Create a rank-1 Tensor with incremental elements in the range [\p start,\p end) with given
  step-size \p step between elements.
  @details The elements are \p start, \p start + \p step, \p start + 2 * \p step, ... The last
  element is the largest one that is smaller than \p end.
  @param[in] start start value of the range.
  @param[in] end end value of the range (exclusive).
  @param[in] step step-size between subsequent elements in the range.
  @param[in] dtype the data type of the Tensor, see cytnx::Type for more information.
  @param[in] device the device type of the Tensor, see cytnx::Device for more information.
  @return
      [Tensor]
  */
  Tensor arange(const cytnx_double &start, const cytnx_double &end, const cytnx_double &step = 1,
                const unsigned int &dtype = Type.Double, const int &device = Device.cpu);
  //@}

  Tensor linspace(const cytnx_double &start, const cytnx_double &end, const cytnx_uint64 &Nelem,
                  const bool &endpoint = true, const unsigned int &dtype = Type.Double,
                  const int &device = Device.cpu);

  // Tensor rand(const std::vector<cytnx_uint64> &size);
  // Tensor randn(const std::vector<cytnx_uint64> &size);
  // Tensor normal(const cytnx_double &mean, cytnx_double &std);

}  // namespace cytnx

#endif  // BACKEND_TORCH

#endif  // CYTNX_GENERATOR_H_
