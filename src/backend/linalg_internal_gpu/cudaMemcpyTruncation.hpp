#ifndef __cudaMemcpyTruncation_internal_H__
#define __cudaMemcpyTruncation_internal_H__

#include <iostream>
#include <vector>

#include "Tensor.hpp"
#include "Type.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace linalg_internal {

#ifdef UNI_GPU
    /// cuSvd
    void cudaMemcpyTruncation_cd(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                                 const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                                 const bool &is_vT, const unsigned int &return_err,
                                 const unsigned int &mindim);
    void cudaMemcpyTruncation_cf(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                                 const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                                 const bool &is_vT, const unsigned int &return_err,
                                 const unsigned int &mindim);
    void cudaMemcpyTruncation_d(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                                const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                                const bool &is_vT, const unsigned int &return_err,
                                const unsigned int &mindim);
    void cudaMemcpyTruncation_f(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                                const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                                const bool &is_vT, const unsigned int &return_err,
                                const unsigned int &mindim);
#endif

  }  // namespace linalg_internal
}  // namespace cytnx

#endif
