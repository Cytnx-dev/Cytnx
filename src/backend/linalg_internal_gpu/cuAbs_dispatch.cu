#include "cuAbs_internal.hpp"
#include "cuUnaryDispatch.cuh"

// #1003 step 11: typed GPU dispatch for the elementwise Abs, replacing the legacy
// lii.cuAbs_ii[dtype] lookup table and its cuDoubleComplex / cuCabs kernels. The kernel, launch
// configuration and dtype dispatch are shared with Exp/Pow/Conj in cuUnaryDispatch.cuh; only the
// operation (AbsOp) and the output rule Abs(complex) -> real (AbsTraits) are specific here.

namespace cytnx {
  namespace linalg_internal {

    void cuAbs_dispatch(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &in, cytnx_uint64 Nelem) {
      cuUnaryDispatch<gpu_unary::AbsTraits>(out, in, Nelem, gpu_unary::AbsOp{}, "cuAbs_dispatch");
    }

  }  // namespace linalg_internal
}  // namespace cytnx
