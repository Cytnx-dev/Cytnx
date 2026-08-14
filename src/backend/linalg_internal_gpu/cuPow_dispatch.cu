#include "cuPow_internal.hpp"
#include "cuUnaryDispatch.cuh"

// #1003 step 11: typed GPU dispatch for elementwise Pow(., p), replacing the legacy
// lii.cuPow_ii[dtype] lookup table and its CUDA C complex kernels. Pow is dtype-preserving: the
// caller pre-casts the input to a floating/complex dispatch dtype (integer/bool inputs are
// promoted to Double first) and passes it as both `out` and `in`. Kernel and dispatch are shared
// -- see cuUnaryDispatch.cuh; PowOp carries the double exponent.

namespace cytnx {
  namespace linalg_internal {

    void cuPow_dispatch(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &in, cytnx_uint64 Nelem,
                        double p) {
      cuUnaryDispatch<gpu_unary::FloatingTraits>(out, in, Nelem, gpu_unary::PowOp{p},
                                                 "cuPow_dispatch");
    }

  }  // namespace linalg_internal
}  // namespace cytnx
