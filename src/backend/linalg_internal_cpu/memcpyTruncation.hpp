#ifndef __memcpyTruncation_internal_H__
#define __memcpyTruncation_internal_H__

#include <iostream>
#include <vector>

#include "Tensor.hpp"
#include "Type.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace linalg_internal {

    void memcpyTruncation_cd(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                             const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                             const bool &is_vT, const unsigned int &return_err,
                             const unsigned int &mindim);
    void memcpyTruncation_cf(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                             const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                             const bool &is_vT, const unsigned int &return_err,
                             const unsigned int &mindim);
    void memcpyTruncation_d(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                            const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                            const bool &is_vT, const unsigned int &return_err,
                            const unsigned int &mindim);
    void memcpyTruncation_f(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                            const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                            const bool &is_vT, const unsigned int &return_err,
                            const unsigned int &mindim);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif
