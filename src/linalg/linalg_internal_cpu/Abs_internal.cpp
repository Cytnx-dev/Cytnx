#include "linalg/linalg_internal_cpu/Abs_internal.hpp"
#include "cytnx_error.hpp"
//#include "utils/lapack_wrapper.hpp"

#ifdef UNI_OMP
  #include <omp.h>
#endif

// stackoverflow.com/questions/33738509/whats-the-difference-between-abs-and-fabs

namespace cytnx {
  namespace linalg_internal {
    void Abs_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_complex128 *_ten = (cytnx_complex128 *)ten->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = std::abs(_ten[n]);
      }
    }

    void Abs_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_complex64 *_ten = (cytnx_complex64 *)ten->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = std::abs(_ten[n]);
      }
    }

    void Abs_internal_d(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_ten = (cytnx_double *)ten->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = std::abs(_ten[n]);
      }
    }

    void Abs_internal_f(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_ten = (cytnx_float *)ten->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = std::abs(_ten[n]);
      }
    }

    void Abs_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten,
                          const cytnx_uint64 &Nelem) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_ten = (cytnx_int64 *)ten->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = std::abs(_ten[n]);
      }
    }

    void Abs_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten,
                          const cytnx_uint64 &Nelem) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_int32 *_ten = (cytnx_int32 *)ten->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = std::abs(_ten[n]);
      }
    }

    void Abs_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten,
                          const cytnx_uint64 &Nelem) {
      cytnx_int16 *_out = (cytnx_int16 *)out->Mem;
      cytnx_int16 *_ten = (cytnx_int16 *)ten->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = std::abs(cytnx_double(_ten[n]));
      }
    }

    void Abs_internal_pass(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten,
                           const cytnx_uint64 &Nelem) {
      cytnx_error_msg(
        true, "[ERROR][Abs_internal] Called pass, which should not be called by frontend.%s", "\n");
    }

  }  // namespace linalg_internal

}  // namespace cytnx
