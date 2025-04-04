#include "Diag_internal.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"

namespace cytnx {
  namespace linalg_internal {

    template <class T>
    void Diag_internal_driver(T *out, T *in, const cytnx_uint64 &L, const cytnx_bool &isrank2) {
      if (isrank2) {
        for (cytnx_uint64 i = 0; i < L; i++) out[i] = in[i * L + i];
      } else {
        for (cytnx_uint64 i = 0; i < L; i++) out[i * L + i] = in[i];
      }
    }

    void Diag_internal_b(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                         const cytnx_bool &isrank2) {
      cytnx_bool *_out = (cytnx_bool *)out->data();
      cytnx_bool *_ten = (cytnx_bool *)ten->data();

      Diag_internal_driver(_out, _ten, L, isrank2);
    }

    void Diag_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const cytnx_bool &isrank2) {
      cytnx_int16 *_out = (cytnx_int16 *)out->data();
      cytnx_int16 *_ten = (cytnx_int16 *)ten->data();

      Diag_internal_driver(_out, _ten, L, isrank2);
    }

    void Diag_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const cytnx_bool &isrank2) {
      cytnx_uint16 *_out = (cytnx_uint16 *)out->data();
      cytnx_uint16 *_ten = (cytnx_uint16 *)ten->data();

      Diag_internal_driver(_out, _ten, L, isrank2);
    }

    void Diag_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const cytnx_bool &isrank2) {
      cytnx_int32 *_out = (cytnx_int32 *)out->data();
      cytnx_int32 *_ten = (cytnx_int32 *)ten->data();

      Diag_internal_driver(_out, _ten, L, isrank2);
    }

    void Diag_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const cytnx_bool &isrank2) {
      cytnx_uint32 *_out = (cytnx_uint32 *)out->data();
      cytnx_uint32 *_ten = (cytnx_uint32 *)ten->data();

      Diag_internal_driver(_out, _ten, L, isrank2);
    }

    void Diag_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const cytnx_bool &isrank2) {
      cytnx_int64 *_out = (cytnx_int64 *)out->data();
      cytnx_int64 *_ten = (cytnx_int64 *)ten->data();

      Diag_internal_driver(_out, _ten, L, isrank2);
    }

    void Diag_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const cytnx_bool &isrank2) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->data();
      cytnx_uint64 *_ten = (cytnx_uint64 *)ten->data();

      Diag_internal_driver(_out, _ten, L, isrank2);
    }

    void Diag_internal_d(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                         const cytnx_bool &isrank2) {
      cytnx_double *_out = (cytnx_double *)out->data();
      cytnx_double *_ten = (cytnx_double *)ten->data();

      Diag_internal_driver(_out, _ten, L, isrank2);
    }

    void Diag_internal_f(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                         const cytnx_bool &isrank2) {
      cytnx_float *_out = (cytnx_float *)out->data();
      cytnx_float *_ten = (cytnx_float *)ten->data();

      Diag_internal_driver(_out, _ten, L, isrank2);
    }

    void Diag_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                          const cytnx_bool &isrank2) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->data();
      cytnx_complex128 *_ten = (cytnx_complex128 *)ten->data();

      Diag_internal_driver(_out, _ten, L, isrank2);
    }

    void Diag_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                          const cytnx_bool &isrank2) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->data();
      cytnx_complex64 *_ten = (cytnx_complex64 *)ten->data();

      Diag_internal_driver(_out, _ten, L, isrank2);
    }

  }  // namespace linalg_internal

}  // namespace cytnx
