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

    template <class T>
    void _trace_2d_gpu(Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag) {
      cytnx::UniTensor I_UT = cytnx::UniTensor::eye(Ndiag, {}, true, Tn.dtype(), Tn.device());
      // similar to _trace_nd_gpu
      UniTensor UTn = UniTensor(Tn, false, 2);
      I_UT.relabel_({UTn._impl->_labels[0], UTn._impl->_labels[1]});
      out = Contract(I_UT, UTn).get_block_();
    }

    template <class T>
    void _trace_nd_gpu(Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag, cytnx_uint64 Nelem,
                       const std::vector<cytnx_uint64> &accu,
                       const std::vector<cytnx_uint64> &remain_rank_id,
                       const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      // currently identical to CPU version
      cytnx::UniTensor I_UT = cytnx::UniTensor::eye(Ndiag, {}, true, Tn.dtype(), Tn.device());

      UniTensor UTn = UniTensor(Tn, false, 2);
      I_UT.relabel_({UTn._impl->_labels[ax1], UTn._impl->_labels[ax2]});

      out = Contract(I_UT, UTn).get_block_();
    }

    void cuTrace_internal_cd(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                             cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                             const std::vector<cytnx_uint64> &remain_rank_id,
                             const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                             cytnx_uint64 ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_complex128>(out, Tn, Ndiag);
      } else {
        _trace_nd_gpu<cytnx_complex128>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1,
                                        ax2);
      }
    }

    void cuTrace_internal_cf(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                             cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                             const std::vector<cytnx_uint64> &remain_rank_id,
                             const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                             cytnx_uint64 ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_complex64>(out, Tn, Ndiag);
      } else {
        _trace_nd_gpu<cytnx_complex64>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1,
                                       ax2);
      }
    }

    void cuTrace_internal_d(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                            cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                            const std::vector<cytnx_uint64> &remain_rank_id,
                            const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                            cytnx_uint64 ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_double>(out, Tn, Ndiag);
      } else {
        _trace_nd_gpu<cytnx_double>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
      }
    }

    void cuTrace_internal_f(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                            cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                            const std::vector<cytnx_uint64> &remain_rank_id,
                            const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                            cytnx_uint64 ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_float>(out, Tn, Ndiag);
      } else {
        _trace_nd_gpu<cytnx_float>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
      }
    }

    void cuTrace_internal_u64(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                              cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                              cytnx_uint64 ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_uint64>(out, Tn, Ndiag);
      } else {
        _trace_nd_gpu<cytnx_uint64>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
      }
    }

    void cuTrace_internal_i64(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                              cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                              cytnx_uint64 ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_int64>(out, Tn, Ndiag);
      } else {
        _trace_nd_gpu<cytnx_int64>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
      }
    }

    void cuTrace_internal_u32(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                              cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                              cytnx_uint64 ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_uint32>(out, Tn, Ndiag);
      } else {
        _trace_nd_gpu<cytnx_uint32>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
      }
    }

    void cuTrace_internal_i32(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                              cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                              cytnx_uint64 ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_int32>(out, Tn, Ndiag);
      } else {
        _trace_nd_gpu<cytnx_int32>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
      }
    }

    void cuTrace_internal_u16(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                              cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                              cytnx_uint64 ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_uint16>(out, Tn, Ndiag);
      } else {
        _trace_nd_gpu<cytnx_uint16>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
      }
    }

    void cuTrace_internal_i16(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                              cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                              cytnx_uint64 ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_int16>(out, Tn, Ndiag);
      } else {
        _trace_nd_gpu<cytnx_int16>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
      }
    }

    void cuTrace_internal_b(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                            cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                            const std::vector<cytnx_uint64> &remain_rank_id,
                            const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                            cytnx_uint64 ax2) {
      cytnx_error_msg(true, "[internal][cuTrace] bool is not available. %s", "\n");
    }

  }  // namespace linalg_internal

}  // namespace cytnx
