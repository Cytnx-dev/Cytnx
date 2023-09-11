#ifndef __cudaMemcpyTruncation_internal_H__
#define __cudaMemcpyTruncation_internal_H__

#include <iostream>
#include <vector>
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "linalg/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg_internal {

#ifdef UNI_GPU
    /// cuSvd
    void cudaMemcpyTruncation_cd(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                                 const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                                 const bool &is_vT, const unsigned int &return_err);
    void cudaMemcpyTruncation_cf(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                                 const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                                 const bool &is_vT, const unsigned int &return_err);
    void cudaMemcpyTruncation_d(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                                const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                                const bool &is_vT, const unsigned int &return_err);
    void cudaMemcpyTruncation_f(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                                const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                                const bool &is_vT, const unsigned int &return_err);
#endif

  }  // namespace linalg_internal
}  // namespace cytnx

#endif
