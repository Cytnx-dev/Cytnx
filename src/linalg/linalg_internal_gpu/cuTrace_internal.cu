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
      T a = 0;
      T *rawdata = Tn.storage().data<T>();
      cytnx_uint64 Ldim = Tn.shape()[1];
      // reduce!
      for (cytnx_uint64 i = 0; i < Ndiag; i++) a += rawdata[i * Ldim + i];
      out.storage().at<T>(0) = a;
    }

    template <class T>
    void _trace_nd_gpu(Tensor &out, const Tensor &Tn, const cytnx_uint64 &Ndiag,
                       const cytnx_uint64 &Nelem, const std::vector<cytnx_uint64> &accu,
                       const std::vector<cytnx_uint64> &remain_rank_id,
                       const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                       const cytnx_uint64 &ax2) {
      cytnx::UniTensor I_UT = cytnx::UniTensor(zeros(Ndiag, Tn.dtype(), Tn.device()), true, -1);

      I_UT.set_labels({"0", "1"});
      UniTensor UTn = UniTensor(Tn, false, 2);
      UTn.set_labels(
        vec_cast<cytnx_uint64, cytnx_int64>(vec_range(100, 100 + UTn.labels().size())));
      UTn._impl->_labels[ax1] = "0";
      UTn._impl->_labels[ax2] = "1";
      out = Contract(I_UT, UTn).get_block_();
    }

    void cuTrace_internal_cd(const bool &is_2d, Tensor &out, const Tensor &Tn,
                             const cytnx_uint64 &Ndiag, const int &Nomp, const cytnx_uint64 &Nelem,
                             const std::vector<cytnx_uint64> &accu,
                             const std::vector<cytnx_uint64> &remain_rank_id,
                             const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                             const cytnx_uint64 &ax2) {
      if (is_2d) {
#ifdef UNI_OMP
        _trace_2d_para<cytnx_complex128>(out, Tn, Ndiag, Nomp);
#else
        _trace_2d<cytnx_complex128>(out, Tn, Ndiag);
#endif
      } else {
#ifdef UNI_OMP
        _trace_nd_para<cytnx_complex128>(out, Tn, Ndiag, Nomp, Nelem, accu, remain_rank_id, shape,
                                         ax1, ax2);
#else
        _trace_nd<cytnx_complex128>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void cuTrace_internal_cf(const bool &is_2d, Tensor &out, const Tensor &Tn,
                             const cytnx_uint64 &Ndiag, const int &Nomp, const cytnx_uint64 &Nelem,
                             const std::vector<cytnx_uint64> &accu,
                             const std::vector<cytnx_uint64> &remain_rank_id,
                             const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                             const cytnx_uint64 &ax2) {
      if (is_2d) {
#ifdef UNI_OMP
        _trace_2d_para<cytnx_complex64>(out, Tn, Ndiag, Nomp);
#else
        _trace_2d<cytnx_complex64>(out, Tn, Ndiag);
#endif
      } else {
#ifdef UNI_OMP
        _trace_nd_para<cytnx_complex64>(out, Tn, Ndiag, Nomp, Nelem, accu, remain_rank_id, shape,
                                        ax1, ax2);
#else
        _trace_nd<cytnx_complex64>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void cuTrace_internal_d(const bool &is_2d, Tensor &out, const Tensor &Tn,
                            const cytnx_uint64 &Ndiag, const int &Nomp, const cytnx_uint64 &Nelem,
                            const std::vector<cytnx_uint64> &accu,
                            const std::vector<cytnx_uint64> &remain_rank_id,
                            const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                            const cytnx_uint64 &ax2) {
      if (is_2d) {
#ifdef UNI_OMP
        _trace_2d_para<cytnx_double>(out, Tn, Ndiag, Nomp);
#else
        _trace_2d<cytnx_double>(out, Tn, Ndiag);
#endif
      } else {
#ifdef UNI_OMP
        _trace_nd_para<cytnx_double>(out, Tn, Ndiag, Nomp, Nelem, accu, remain_rank_id, shape, ax1,
                                     ax2);
#else
        _trace_nd<cytnx_double>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void cuTrace_internal_f(const bool &is_2d, Tensor &out, const Tensor &Tn,
                            const cytnx_uint64 &Ndiag, const int &Nomp, const cytnx_uint64 &Nelem,
                            const std::vector<cytnx_uint64> &accu,
                            const std::vector<cytnx_uint64> &remain_rank_id,
                            const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                            const cytnx_uint64 &ax2) {
      if (is_2d) {
#ifdef UNI_OMP
        _trace_2d_para<cytnx_float>(out, Tn, Ndiag, Nomp);
#else
        _trace_2d<cytnx_float>(out, Tn, Ndiag);
#endif
      } else {
#ifdef UNI_OMP
        _trace_nd_para<cytnx_float>(out, Tn, Ndiag, Nomp, Nelem, accu, remain_rank_id, shape, ax1,
                                    ax2);
#else
        _trace_nd<cytnx_float>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void cuTrace_internal_u64(const bool &is_2d, Tensor &out, const Tensor &Tn,
                              const cytnx_uint64 &Ndiag, const int &Nomp, const cytnx_uint64 &Nelem,
                              const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2) {
      if (is_2d) {
#ifdef UNI_OMP
        _trace_2d_para<cytnx_uint64>(out, Tn, Ndiag, Nomp);
#else
        _trace_2d<cytnx_uint64>(out, Tn, Ndiag);
#endif
      } else {
#ifdef UNI_OMP
        _trace_nd_para<cytnx_uint64>(out, Tn, Ndiag, Nomp, Nelem, accu, remain_rank_id, shape, ax1,
                                     ax2);
#else
        _trace_nd<cytnx_uint64>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void cuTrace_internal_i64(const bool &is_2d, Tensor &out, const Tensor &tn,
                              const cytnx_uint64 &ndiag, const int &nomp, const cytnx_uint64 &nelem,
                              const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2) {
      if (is_2d) {
#ifdef UNI_OMP
        _trace_2d_para<cytnx_int64>(out, tn, ndiag, nomp);
#else
        _trace_2d<cytnx_int64>(out, tn, ndiag);
#endif
      } else {
#ifdef UNI_OMP
        _trace_nd_para<cytnx_int64>(out, tn, ndiag, nomp, nelem, accu, remain_rank_id, shape, ax1,
                                    ax2);
#else
        _trace_nd<cytnx_int64>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void cuTrace_internal_u32(const bool &is_2d, Tensor &out, const Tensor &tn,
                              const cytnx_uint64 &ndiag, const int &nomp, const cytnx_uint64 &nelem,
                              const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2) {
      if (is_2d) {
#ifdef UNI_OMP
        _trace_2d_para<cytnx_uint32>(out, tn, ndiag, nomp);
#else
        _trace_2d<cytnx_uint32>(out, tn, ndiag);
#endif
      } else {
#ifdef UNI_OMP
        _trace_nd_para<cytnx_uint32>(out, tn, ndiag, nomp, nelem, accu, remain_rank_id, shape, ax1,
                                     ax2);
#else
        _trace_nd<cytnx_uint32>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void cuTrace_internal_i32(const bool &is_2d, Tensor &out, const Tensor &tn,
                              const cytnx_uint64 &ndiag, const int &nomp, const cytnx_uint64 &nelem,
                              const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2) {
      if (is_2d) {
#ifdef UNI_OMP
        _trace_2d_para<cytnx_int32>(out, tn, ndiag, nomp);
#else
        _trace_2d<cytnx_int32>(out, tn, ndiag);
#endif
      } else {
#ifdef UNI_OMP
        _trace_nd_para<cytnx_int32>(out, tn, ndiag, nomp, nelem, accu, remain_rank_id, shape, ax1,
                                    ax2);
#else
        _trace_nd<cytnx_int32>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void cuTrace_internal_u16(const bool &is_2d, Tensor &out, const Tensor &tn,
                              const cytnx_uint64 &ndiag, const int &nomp, const cytnx_uint64 &nelem,
                              const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2) {
      if (is_2d) {
#ifdef UNI_OMP
        _trace_2d_para<cytnx_uint16>(out, tn, ndiag, nomp);
#else
        _trace_2d<cytnx_uint16>(out, tn, ndiag);
#endif
      } else {
#ifdef UNI_OMP
        _trace_nd_para<cytnx_uint16>(out, tn, ndiag, nomp, nelem, accu, remain_rank_id, shape, ax1,
                                     ax2);
#else
        _trace_nd<cytnx_uint16>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void cuTrace_internal_i16(const bool &is_2d, Tensor &out, const Tensor &tn,
                              const cytnx_uint64 &ndiag, const int &nomp, const cytnx_uint64 &nelem,
                              const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2) {
      if (is_2d) {
#ifdef UNI_OMP
        _trace_2d_para<cytnx_int16>(out, tn, ndiag, nomp);
#else
        _trace_2d<cytnx_int16>(out, tn, ndiag);
#endif
      } else {
#ifdef UNI_OMP
        _trace_nd_para<cytnx_int16>(out, tn, ndiag, nomp, nelem, accu, remain_rank_id, shape, ax1,
                                    ax2);
#else
        _trace_nd<cytnx_int16>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void cuTrace_internal_b(const bool &is_2d, Tensor &out, const Tensor &tn,
                            const cytnx_uint64 &ndiag, const int &nomp, const cytnx_uint64 &nelem,
                            const std::vector<cytnx_uint64> &accu,
                            const std::vector<cytnx_uint64> &remain_rank_id,
                            const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                            const cytnx_uint64 &ax2) {
      cytnx_error_msg(true, "[internal][cuTrace] bool is not available. %s", "\n");
    }

  }  // namespace linalg_internal

}  // namespace cytnx
