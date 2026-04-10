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
    void _trace_2d_gpu(Tensor &out, const Tensor &Tn, const cytnx_uint64 &Ndiag) {
      cytnx::UniTensor I_UT = cytnx::UniTensor::eye(Ndiag, {}, true, Tn.dtype(), Tn.device());
      // similar to _trace_nd_gpu
      UniTensor UTn = UniTensor(Tn, false, 2);
      I_UT.set_labels({UTn._impl->_labels[0], UTn._impl->_labels[1]});
      out = Contract(I_UT, UTn).get_block_();
    }

    template <class T>
    void _trace_nd_gpu(Tensor &out, const Tensor &Tn, const cytnx_uint64 &Ndiag,
                       const cytnx_uint64 &Nelem, const std::vector<cytnx_uint64> &accu,
                       const std::vector<cytnx_uint64> &remain_rank_id,
                       const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                       const cytnx_uint64 &ax2) {
      // currently identical to CPU version
      cytnx::UniTensor I_UT = cytnx::UniTensor::eye(Ndiag, {}, true, Tn.dtype(), Tn.device());

      UniTensor UTn = UniTensor(Tn, false, 2);
      I_UT.set_labels({UTn._impl->_labels[ax1], UTn._impl->_labels[ax2]});

      out = Contract(I_UT, UTn).get_block_();
    }

    void cuTrace_internal_cd(const bool &is_2d, Tensor &out, const Tensor &Tn,
                             const cytnx_uint64 &Ndiag, const cytnx_uint64 &Nelem,
                             const std::vector<cytnx_uint64> &accu,
                             const std::vector<cytnx_uint64> &remain_rank_id,
                             const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                             const cytnx_uint64 &ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_complex128>(out, Tn, Ndiag);
      } else {
        _trace_nd_gpu<cytnx_complex128>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1,
                                        ax2);
      }
    }

    void cuTrace_internal_cf(const bool &is_2d, Tensor &out, const Tensor &Tn,
                             const cytnx_uint64 &Ndiag, const cytnx_uint64 &Nelem,
                             const std::vector<cytnx_uint64> &accu,
                             const std::vector<cytnx_uint64> &remain_rank_id,
                             const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                             const cytnx_uint64 &ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_complex64>(out, Tn, Ndiag);
      } else {
        _trace_nd_gpu<cytnx_complex64>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1,
                                       ax2);
      }
    }

    void cuTrace_internal_d(const bool &is_2d, Tensor &out, const Tensor &Tn,
                            const cytnx_uint64 &Ndiag, const cytnx_uint64 &Nelem,
                            const std::vector<cytnx_uint64> &accu,
                            const std::vector<cytnx_uint64> &remain_rank_id,
                            const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                            const cytnx_uint64 &ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_double>(out, Tn, Ndiag);
      } else {
        _trace_nd_gpu<cytnx_double>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
      }
    }

    void cuTrace_internal_f(const bool &is_2d, Tensor &out, const Tensor &Tn,
                            const cytnx_uint64 &Ndiag, const cytnx_uint64 &Nelem,
                            const std::vector<cytnx_uint64> &accu,
                            const std::vector<cytnx_uint64> &remain_rank_id,
                            const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                            const cytnx_uint64 &ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_float>(out, Tn, Ndiag);
      } else {
        _trace_nd_gpu<cytnx_float>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
      }
    }

    void cuTrace_internal_u64(const bool &is_2d, Tensor &out, const Tensor &Tn,
                              const cytnx_uint64 &Ndiag, const cytnx_uint64 &Nelem,
                              const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_uint64>(out, Tn, Ndiag);
      } else {
        _trace_nd_gpu<cytnx_uint64>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
      }
    }

    void cuTrace_internal_i64(const bool &is_2d, Tensor &out, const Tensor &tn,
                              const cytnx_uint64 &ndiag, const cytnx_uint64 &nelem,
                              const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_int64>(out, tn, ndiag);
      } else {
        _trace_nd_gpu<cytnx_int64>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
      }
    }

    void cuTrace_internal_u32(const bool &is_2d, Tensor &out, const Tensor &tn,
                              const cytnx_uint64 &ndiag, const cytnx_uint64 &nelem,
                              const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_uint32>(out, tn, ndiag);
      } else {
        _trace_nd_gpu<cytnx_uint32>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
      }
    }

    void cuTrace_internal_i32(const bool &is_2d, Tensor &out, const Tensor &tn,
                              const cytnx_uint64 &ndiag, const cytnx_uint64 &nelem,
                              const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_int32>(out, tn, ndiag);
      } else {
        _trace_nd_gpu<cytnx_int32>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
      }
    }

    void cuTrace_internal_u16(const bool &is_2d, Tensor &out, const Tensor &tn,
                              const cytnx_uint64 &ndiag, const cytnx_uint64 &nelem,
                              const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_uint16>(out, tn, ndiag);
      } else {
        _trace_nd_gpu<cytnx_uint16>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
      }
    }

    void cuTrace_internal_i16(const bool &is_2d, Tensor &out, const Tensor &tn,
                              const cytnx_uint64 &ndiag, const cytnx_uint64 &nelem,
                              const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2) {
      if (is_2d) {
        _trace_2d_gpu<cytnx_int16>(out, tn, ndiag);
      } else {
        _trace_nd_gpu<cytnx_int16>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
      }
    }

    void cuTrace_internal_b(const bool &is_2d, Tensor &out, const Tensor &tn,
                            const cytnx_uint64 &ndiag, const cytnx_uint64 &nelem,
                            const std::vector<cytnx_uint64> &accu,
                            const std::vector<cytnx_uint64> &remain_rank_id,
                            const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                            const cytnx_uint64 &ax2) {
      cytnx_error_msg(true, "[internal][cuTrace] bool is not available. %s", "\n");
    }

  }  // namespace linalg_internal

}  // namespace cytnx
