#include "linalg/linalg.hpp"
#include "Accessor.hpp"
#include <iostream>

namespace cytnx {
  namespace linalg {
    typedef Accessor ac;
    std::vector<Tensor> Svd_truncate(const Tensor &Tin, const cytnx_uint64 &keepdim,
                                     const bool &is_U, const bool &is_vT) {
      std::vector<Tensor> tmps = Svd(Tin, is_U, is_vT);

      cytnx_uint64 id = 0;
      cytnx_error_msg(tmps[0].shape()[0] < keepdim,
                      "[ERROR] keepdim should be <= the valid # of singular value, %d!\n",
                      tmps[0].shape()[0]);

      tmps[id] = tmps[id].get({ac::range(0, keepdim)});

      if (is_U) {
        id++;
        tmps[id] = tmps[id].get({ac::all(), ac::range(0, keepdim)});
      }
      if (is_vT) {
        id++;
        tmps[id] = tmps[id].get({ac::range(0, keepdim), ac::all()});
      }
      return tmps;
    }
  }  // namespace linalg
}  // namespace cytnx
