#include "linalg/linalg_internal_cpu/Diag_internal.hpp"
#include "cytnx_error.hpp"
#include "utils/lapack_wrapper.hpp"

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {
  namespace linalg_internal {

    template <class T>
    void Diag_internal_driver(T *out, T *in, const cytnx_uint64 &L) {
#ifdef UNI_OMP
  #pragma omp parallel for
#endif
      for (cytnx_uint64 i = 0; i < L; i++) {
        out[i * L + i] = in[i];
      }
    }

    void Diag_internal_b(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L) {
      cytnx_bool *_out = (cytnx_bool *)out->Mem;
      cytnx_bool *_ten = (cytnx_bool *)ten->Mem;

      Diag_internal_driver(_out, _ten, L);
    }

    void Diag_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L) {
      cytnx_int16 *_out = (cytnx_int16 *)out->Mem;
      cytnx_int16 *_ten = (cytnx_int16 *)ten->Mem;

      Diag_internal_driver(_out, _ten, L);
    }

    void Diag_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L) {
      cytnx_uint16 *_out = (cytnx_uint16 *)out->Mem;
      cytnx_uint16 *_ten = (cytnx_uint16 *)ten->Mem;

      Diag_internal_driver(_out, _ten, L);
    }

    void Diag_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_int32 *_ten = (cytnx_int32 *)ten->Mem;

      Diag_internal_driver(_out, _ten, L);
    }

    void Diag_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L) {
      cytnx_uint32 *_out = (cytnx_uint32 *)out->Mem;
      cytnx_uint32 *_ten = (cytnx_uint32 *)ten->Mem;

      Diag_internal_driver(_out, _ten, L);
    }

    void Diag_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_ten = (cytnx_int64 *)ten->Mem;

      Diag_internal_driver(_out, _ten, L);
    }

    void Diag_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_ten = (cytnx_uint64 *)ten->Mem;

      Diag_internal_driver(_out, _ten, L);
    }

    void Diag_internal_d(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_ten = (cytnx_double *)ten->Mem;

      Diag_internal_driver(_out, _ten, L);
    }

    void Diag_internal_f(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_ten = (cytnx_float *)ten->Mem;

      Diag_internal_driver(_out, _ten, L);
    }

    void Diag_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->Mem;
      cytnx_complex128 *_ten = (cytnx_complex128 *)ten->Mem;

      Diag_internal_driver(_out, _ten, L);
    }

    void Diag_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->Mem;
      cytnx_complex64 *_ten = (cytnx_complex64 *)ten->Mem;

      Diag_internal_driver(_out, _ten, L);
    }

  }  // namespace linalg_internal

}  // namespace cytnx
