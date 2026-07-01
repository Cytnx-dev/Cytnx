#include "cuTrace_internal.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"
#include "utils/cucomplex_arithmetic.hpp"

#include "Generator.hpp"
#include "utils/utils.hpp"

#include "UniTensor.hpp"
#include <vector>

namespace cytnx {
  namespace linalg_internal {
    namespace {

      template <class T>
      Tensor _trace_2d_gpu(const Tensor &Tn, cytnx_uint64 Ndiag) {
        cytnx::UniTensor I_UT = cytnx::UniTensor::eye(Ndiag, {}, true, Tn.dtype(), Tn.device());
        UniTensor UTn = UniTensor(Tn, false, 2);
        I_UT.relabel_({UTn._impl->_labels[0], UTn._impl->_labels[1]});
        return Contract(I_UT, UTn).get_block_();
      }

      template <class T>
      Tensor _trace_nd_gpu(const Tensor &Tn, cytnx_uint64 Ndiag, cytnx_uint64 ax1,
                           cytnx_uint64 ax2) {
        cytnx::UniTensor I_UT = cytnx::UniTensor::eye(Ndiag, {}, true, Tn.dtype(), Tn.device());
        UniTensor UTn = UniTensor(Tn, false, 2);
        I_UT.relabel_({UTn._impl->_labels[ax1], UTn._impl->_labels[ax2]});
        return Contract(I_UT, UTn).get_block_();
      }

      // Dispatches to the rank-2 or rank-N helper. The two helpers existed before
      // the dispatcher API was simplified to (Tn, ax1, ax2); this trampoline
      // derives is_2d / Ndiag from the input so callers do not have to.
      template <class T>
      Tensor TraceDispatchGpu(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
        const cytnx_uint64 Ndiag = Tn.shape()[ax1];
        if (Tn.shape().size() == 2) return _trace_2d_gpu<T>(Tn, Ndiag);
        return _trace_nd_gpu<T>(Tn, Ndiag, ax1, ax2);
      }

    }  // namespace

    Tensor cuTrace_internal_cd(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceDispatchGpu<cytnx_complex128>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_cf(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceDispatchGpu<cytnx_complex64>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_d(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceDispatchGpu<cytnx_double>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_f(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceDispatchGpu<cytnx_float>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_u64(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceDispatchGpu<cytnx_uint64>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_i64(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceDispatchGpu<cytnx_int64>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_u32(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceDispatchGpu<cytnx_uint32>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_i32(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceDispatchGpu<cytnx_int32>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_u16(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceDispatchGpu<cytnx_uint16>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_i16(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceDispatchGpu<cytnx_int16>(Tn, ax1, ax2);
    }

    Tensor cuTrace_internal_b(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      cytnx_error_msg(true, "[internal][cuTrace] bool is not available. %s", "\n");
      return Tensor();
    }

  }  // namespace linalg_internal

}  // namespace cytnx
