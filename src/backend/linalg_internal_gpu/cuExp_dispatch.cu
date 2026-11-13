#include "cuExp_internal.hpp"
#include "cuUnaryDispatch.cuh"

// #1003 step 11: typed GPU dispatch for elementwise Exp, replacing the legacy lii.cuExp_ii[dtype]
// lookup table and its CUDA C complex kernels. Exp is dtype-preserving: the caller pre-casts the
// input to a floating/complex dispatch dtype (integer/bool inputs are promoted to Double first)
// and passes it as both `out` and `in`. Kernel and dispatch are shared -- see cuUnaryDispatch.cuh.

namespace cytnx {
  namespace linalg_internal {

    void cuExp_dispatch(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &in, cytnx_uint64 Nelem) {
      cuUnaryDispatch<gpu_unary::FloatingTraits>(out, in, Nelem, gpu_unary::ExpOp{},
                                                 "cuExp_dispatch");
    }

  }  // namespace linalg_internal
}  // namespace cytnx
