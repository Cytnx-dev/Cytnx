#include "Trace_internal.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"

#include "Generator.hpp"
#include "utils/utils.hpp"

#include "UniTensor.hpp"
#include <vector>

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {
  namespace linalg_internal {

    template <class T>
    void _trace_2d(Tensor &out, const Tensor &Tn, const cytnx_uint64 &Ndiag) {
      T a = 0;
      T *rawdata = Tn.storage().data<T>();
      cytnx_uint64 Ldim = Tn.shape()[1];
      for (cytnx_uint64 i = 0; i < Ndiag; i++) a += rawdata[i * Ldim + i];
      out.storage().at<T>(0) = a;
    }

    template <class T>
    void _trace_nd(Tensor &out, const Tensor &Tn, const cytnx_uint64 &Ndiag,
                   const cytnx_uint64 &Nelem, const std::vector<cytnx_uint64> &accu,
                   const std::vector<cytnx_uint64> &remain_rank_id,
                   const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                   const cytnx_uint64 &ax2) {
      cytnx::UniTensor I_UT = cytnx::UniTensor(eye(Ndiag, Tn.dtype()), false, -1);

      UniTensor UTn = UniTensor(Tn, false, 2);
      I_UT.set_labels({UTn._impl->_labels[ax1], UTn._impl->_labels[ax2]});

      out = Contract(I_UT, UTn).get_block_();

      // std::vector<cytnx_uint64> indexer(Tn.shape().size(), 0);
      // cytnx_uint64 tmp;
      // for (cytnx_uint64 i = 0; i < Nelem; i++) {
      // tmp = i;
      // // calculate indexer
      // for (int x = 0; x < shape.size(); x++) {
      // indexer[remain_rank_id[x]] = cytnx_uint64(tmp / accu[x]);
      // tmp %= accu[x];
      // }

      // for (cytnx_uint64 d = 0; d < Ndiag; d++) {
      // indexer[ax1] = indexer[ax2] = d;
      // out.storage().at<T>(i) += Tn.at<T>(indexer);
      // }
      // }
    }

// parallel version:
#ifdef UNI_OMP
    template <class T>
    void _trace_2d_para(Tensor &out, const Tensor &Tn, const cytnx_uint64 &Ndiag, const int &Nomp) {
      T a = 0;
      std::vector<T> buffer(Nomp);

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 i = 0; i < Ndiag; i++) buffer[omp_get_thread_num()] += Tn.at<T>({i, i});

      for (int i = 1; i < Nomp; i++) buffer[0] += buffer[i];
      out.storage().at<T>({0}) = buffer[0];
    }

    /*
    template <class T>
    void _trace_nd_para(Tensor &out, const Tensor &Tn, const cytnx_uint64 &Ndiag, const int &Nomp,
                        const cytnx_uint64 &Nelem, const std::vector<cytnx_uint64> &accu,
                        const std::vector<cytnx_uint64> &remain_rank_id, const
    std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1, const cytnx_uint64 &ax2) {
      // decide parallel Nelem or Ndiag:
      if (false and Nelem < Ndiag) {
        // each thread need it's own indexer:
        std::vector<std::vector<cytnx_uint64>> indexers(Nomp,
    std::vector<cytnx_uint64>(Tn.shape().size(), 0));

        #pragma omp parallel for schedule(dynamic)
        for (cytnx_uint64 i = 0; i < Nelem; i++) {
          cytnx_uint64 tmp = i;
          // calculate indexer
          for (int x = 0; x < shape.size(); x++) {
            indexers[omp_get_thread_num()][remain_rank_id[x]] = cytnx_uint64(tmp / accu[x]);
            tmp %= accu[x];
          }

          for (cytnx_uint64 d = 0; d < Ndiag; d++) {
            indexers[omp_get_thread_num()][ax1] = indexers[omp_get_thread_num()][ax2] = d;
            out.storage().at<T>(i) += Tn.at<T>(indexers[omp_get_thread_num()]);
          }
        }

      } else {
        #pragma omp parallel
        {
          std::vector<cytnx_uint64> indexers(Tn.shape().size(), 0);

          #pragma omp for schedule(static)
          for (cytnx_uint64 i = 0; i < Nelem; i++) {
            cytnx_uint64 tmp;
            tmp = i;
            // calculate indexer
            for (int x = 0; x < shape.size(); x++) {
              indexers[remain_rank_id[x]] = cytnx_uint64(tmp / accu[x]);
              tmp %= accu[x];
            }
            std::cout << omp_get_thread_num() << " " << i << Nelem << std::endl;
            for (cytnx_uint64 d = 0; d < Ndiag; d++) {
              indexers[ax1] = indexers[ax2] = d;
              out.storage().at<T>(i) += Tn.at<T>(indexers);
            }
          }// end omp for

        }// end omp para
      }// end if switch
    }
    [Deprecated] */
#endif

    void Trace_internal_cd(const bool &is_2d, Tensor &out, const Tensor &Tn,
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
        _trace_nd<cytnx_complex128>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
#else
        _trace_nd<cytnx_complex128>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void Trace_internal_cf(const bool &is_2d, Tensor &out, const Tensor &Tn,
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
        _trace_nd<cytnx_complex64>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
#else
        _trace_nd<cytnx_complex64>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void Trace_internal_d(const bool &is_2d, Tensor &out, const Tensor &Tn,
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
        _trace_nd<cytnx_double>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
#else
        _trace_nd<cytnx_double>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void Trace_internal_f(const bool &is_2d, Tensor &out, const Tensor &Tn,
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
        _trace_nd<cytnx_float>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
#else
        _trace_nd<cytnx_float>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void Trace_internal_u64(const bool &is_2d, Tensor &out, const Tensor &Tn,
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
        _trace_nd<cytnx_uint64>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
#else
        _trace_nd<cytnx_uint64>(out, Tn, Ndiag, Nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void Trace_internal_i64(const bool &is_2d, Tensor &out, const Tensor &tn,
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
        _trace_nd<cytnx_int64>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
#else
        _trace_nd<cytnx_int64>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void Trace_internal_u32(const bool &is_2d, Tensor &out, const Tensor &tn,
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
        _trace_nd<cytnx_uint32>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
#else
        _trace_nd<cytnx_uint32>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void Trace_internal_i32(const bool &is_2d, Tensor &out, const Tensor &tn,
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
        _trace_nd<cytnx_int32>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
#else
        _trace_nd<cytnx_int32>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void Trace_internal_u16(const bool &is_2d, Tensor &out, const Tensor &tn,
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
        _trace_nd<cytnx_uint16>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
#else
        _trace_nd<cytnx_uint16>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void Trace_internal_i16(const bool &is_2d, Tensor &out, const Tensor &tn,
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
        _trace_nd<cytnx_int16>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
#else
        _trace_nd<cytnx_int16>(out, tn, ndiag, nelem, accu, remain_rank_id, shape, ax1, ax2);
#endif
      }
    }

    void Trace_internal_b(const bool &is_2d, Tensor &out, const Tensor &tn,
                          const cytnx_uint64 &ndiag, const int &nomp, const cytnx_uint64 &nelem,
                          const std::vector<cytnx_uint64> &accu,
                          const std::vector<cytnx_uint64> &remain_rank_id,
                          const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                          const cytnx_uint64 &ax2) {
      cytnx_error_msg(true, "[internal][Trace] bool is not available. %s", "\n");
    }

  }  // namespace linalg_internal

}  // namespace cytnx
