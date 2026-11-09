#include "cuConj_inplace_internal.hpp"
#include "cuUnaryDispatch.cuh"

// #1003 step 11: typed GPU dispatch for the in-place elementwise Conj, replacing the legacy
// lii.cuConj_inplace_ii[dtype] lookup table and its CUDA C complex kernels. Only the complex
// dtypes are dispatched here -- real Conj is a no-op the caller skips. Kernel and dispatch are
// shared with Abs/Exp/Pow; see cuUnaryDispatch.cuh.

namespace cytnx {
  namespace linalg_internal {

    void cuConj_inplace_dispatch(boost::intrusive_ptr<Storage_base> &inout, cytnx_uint64 Nelem) {
      cuUnaryDispatchInplace<gpu_unary::ComplexTraits>(inout, Nelem, gpu_unary::ConjOp{},
                                                       "cuConj_inplace_dispatch");
    }

  }  // namespace linalg_internal
}  // namespace cytnx
