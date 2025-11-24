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
  @brief Create a rank-1 Tensor with all elements initialized to zero.
  @param Nelem the number of elements
  @param dtype the dtype of the Tensor. It can be any type defined in \link cytnx::Type cytnx::Type
  \endlink
  @param device the device that the Tensor is put on. It can be any device defined in \link
  cytnx::Device cytnx::Device \endlink

  @return
      [Tensor]

  */
  Tensor zeros(const cytnx_uint64 &Nelem, const unsigned int &dtype = Type.Double,
               const int &device = Device.cpu);
  /**
  @brief Create a Tensor with all elements initialized to zero.
  @param Nelem the shape of the Tensor
  @param dtype the dtype of the Tensor. It can be any type defined in \link cytnx::Type cytnx::Type
  \endlink
  @param device the device that the Tensor is put on. It can be any device defined in \link
  cytnx::Device cytnx::Device \endlink

  @return
      [Tensor]

  */
  Tensor zeros(const std::vector<cytnx_uint64> &Nelem, const unsigned int &dtype = Type.Double,
               const int &device = Device.cpu);
  //@}

  /**
  @brief Create a square rank-2 Tensor with the diagonal initialized to one and all other elements set to zero.
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
  @brief Create a square rank-2 Tensor with the diagonal initialized to one and all other elements set to zero
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
  @brief Create a rank-1 Tensor with all elements initialized to one.
  @param Nelem the number of elements
  @param dtype the dtype of the Tensor. It can be any type defined in \link cytnx::Type cytnx::Type
  \endlink
  @param device the device that the Tensor is put on. It can be any device defined in \link
  cytnx::Device cytnx::Device \endlink

  @return
      [Tensor]

  */
  Tensor ones(const cytnx_uint64 &Nelem, const unsigned int &dtype = Type.Double,
              const int &device = Device.cpu);
  /**
  @brief Create a Tensor with all elements initialized to one.
  @param Nelem the shape of the Tensor
  @param dtype the dtype of the Tensor. It can be any type defined in \link cytnx::Type cytnx::Type
  \endlink
  @param device the device that the Tensor is put on. It can be any device defined in \link
  cytnx::Device cytnx::Device \endlink

  @return
      [Tensor]

  */
  Tensor ones(const std::vector<cytnx_uint64> &Nelem, const unsigned int &dtype = Type.Double,
              const int &device = Device.cpu);
  //@}

  //@{
  /**
  @brief Create a rank-1 Tensor with incremental unsigned integer elements in the range [0,Nelem)
  @param Nelem the number of incremental elements to create.

  @return
      [Tensor]

  */
  Tensor arange(const cytnx_int64 &Nelem);
  /**
  @brief Create an rank-1 Tensor with elements defined in range [start,end) with assigned step-size
  @param start the start value of the range
  @param end the end value of the range
  @param step the step-size of the range
  @param dtype the dtype of the Tensor. It can be any type defined in \link cytnx::Type cytnx::Type
  \endlink
  @param device the device that the Tensor is put on. It can be any device defined in \link
  cytnx::Device cytnx::Device \endlink

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
