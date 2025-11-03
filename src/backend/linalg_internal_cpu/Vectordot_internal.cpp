#include "Vectordot_internal.hpp"
#include "backend/utils_internal_interface.hpp"
#include "utils/utils.hpp"
#include "backend/lapack_wrapper.hpp"
#include <vector>

namespace cytnx {

  namespace linalg_internal {

    void Vectordot_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &Lin,
                               const boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len, const bool &is_conj) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->data();
      cytnx_complex128 *_Lin = (cytnx_complex128 *)Lin->data();
      cytnx_complex128 *_Rin = (cytnx_complex128 *)Rin->data();

      _out[0] = 0;
      unsigned long long remain = len;
      unsigned long long bias = 0;
      blas_int ONE = 1;
      cytnx_complex128 acres;
      if (sizeof(blas_int) == 8) {
        blas_int MAXX = remain;
        if (is_conj)
          zdotc(&acres, &MAXX, &_Lin[bias], &ONE, &_Rin[bias], &ONE);
        else
          zdotu(&acres, &MAXX, &_Lin[bias], &ONE, &_Rin[bias], &ONE);

        _out[0] += acres;
      } else {
        blas_int MAXX = INT_MAX;
        while (remain != 0) {
          MAXX = remain >= INT_MAX ? INT_MAX : remain;

          if (is_conj)
            zdotc(&acres, &MAXX, &_Lin[bias], &ONE, &_Rin[bias], &ONE);
          else
            zdotu(&acres, &MAXX, &_Lin[bias], &ONE, &_Rin[bias], &ONE);

          _out[0] += acres;
          remain -= MAXX;
          bias += MAXX;
        }
      }
    }
    void Vectordot_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &Lin,
                               const boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len, const bool &is_conj) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->data();
      cytnx_complex64 *_Lin = (cytnx_complex64 *)Lin->data();
      cytnx_complex64 *_Rin = (cytnx_complex64 *)Rin->data();
      _out[0] = 0;
      unsigned long long remain = len;
      unsigned long long bias = 0;
      blas_int ONE = 1;
      cytnx_complex64 acres;
      if (sizeof(blas_int) == 8) {
        blas_int MAXX = remain;
        if (is_conj)
          cdotc(&acres, &MAXX, &_Lin[bias], &ONE, &_Rin[bias], &ONE);
        else
          cdotu(&acres, &MAXX, &_Lin[bias], &ONE, &_Rin[bias], &ONE);

        _out[0] += acres;
      } else {
        blas_int MAXX = INT_MAX;
        while (remain != 0) {
          if (remain >= INT_MAX)
            MAXX = INT_MAX;
          else
            MAXX = remain;

          if (is_conj)
            cdotc(&acres, &MAXX, &_Lin[bias], &ONE, &_Rin[bias], &ONE);
          else
            cdotu(&acres, &MAXX, &_Lin[bias], &ONE, &_Rin[bias], &ONE);

          _out[0] += acres;
          remain -= MAXX;
          bias += MAXX;
        }
      }
    }
    void Vectordot_internal_d(boost::intrusive_ptr<Storage_base> &out,
                              const boost::intrusive_ptr<Storage_base> &Lin,
                              const boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const bool &is_conj) {
      cytnx_double *_out = (cytnx_double *)out->data();
      cytnx_double *_Lin = (cytnx_double *)Lin->data();
      cytnx_double *_Rin = (cytnx_double *)Rin->data();

      _out[0] = 0;
      unsigned long long remain = len;
      unsigned long long bias = 0;
      blas_int ONE = 1;
      cytnx_double acres;
      if (sizeof(blas_int) == 8) {
        blas_int MAXX = remain;
        acres = ddot(&MAXX, &_Lin[bias], &ONE, &_Rin[bias], &ONE);
        _out[0] += acres;
      } else {
        blas_int MAXX = INT_MAX;
        while (remain != 0) {
          if (remain >= INT_MAX)
            MAXX = INT_MAX;
          else
            MAXX = remain;

          acres = ddot(&MAXX, &_Lin[bias], &ONE, &_Rin[bias], &ONE);

          _out[0] += acres;
          remain -= MAXX;
          bias += MAXX;
        }
      }
    }
    void Vectordot_internal_f(boost::intrusive_ptr<Storage_base> &out,
                              const boost::intrusive_ptr<Storage_base> &Lin,
                              const boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const bool &is_conj) {
      cytnx_float *_out = (cytnx_float *)out->data();
      cytnx_float *_Lin = (cytnx_float *)Lin->data();
      cytnx_float *_Rin = (cytnx_float *)Rin->data();

      _out[0] = 0;
      unsigned long long remain = len;
      unsigned long long bias = 0;
      blas_int ONE = 1;
      cytnx_float acres;
      if (sizeof(blas_int) == 8) {
        blas_int MAXX = remain;
        acres = sdot(&MAXX, &_Lin[bias], &ONE, &_Rin[bias], &ONE);
        _out[0] += acres;
      } else {
        blas_int MAXX = INT_MAX;
        while (remain != 0) {
          if (remain >= INT_MAX)
            MAXX = INT_MAX;
          else
            MAXX = remain;

          acres = sdot(&MAXX, &_Lin[bias], &ONE, &_Rin[bias], &ONE);

          _out[0] += acres;
          remain -= MAXX;
          bias += MAXX;
        }
      }
    }
    void Vectordot_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &Lin,
                                const boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len, const bool &is_conj) {
      cytnx_int64 *_out = (cytnx_int64 *)out->data();
      cytnx_int64 *_Lin = (cytnx_int64 *)Lin->data();
      cytnx_int64 *_Rin = (cytnx_int64 *)Rin->data();

      _out[0] = 0;
      for (cytnx_uint64 i = 0; i < len; i++) {
        _out[0] += _Lin[i] * _Rin[i];
      }
    }
    void Vectordot_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &Lin,
                                const boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len, const bool &is_conj) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->data();
      cytnx_uint64 *_Lin = (cytnx_uint64 *)Lin->data();
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->data();

      _out[0] = 0;
      for (cytnx_uint64 i = 0; i < len; i++) {
        _out[0] += _Lin[i] * _Rin[i];
      }
    }
    void Vectordot_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &Lin,
                                const boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len, const bool &is_conj) {
      blas_int *_out = (blas_int *)out->data();
      blas_int *_Lin = (blas_int *)Lin->data();
      blas_int *_Rin = (blas_int *)Rin->data();

      _out[0] = 0;
      for (cytnx_uint64 i = 0; i < len; i++) {
        _out[0] += _Lin[i] * _Rin[i];
      }
    }
    void Vectordot_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &Lin,
                                const boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len, const bool &is_conj) {
      cytnx_uint32 *_out = (cytnx_uint32 *)out->data();
      cytnx_uint32 *_Lin = (cytnx_uint32 *)Lin->data();
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->data();

      _out[0] = 0;
      for (cytnx_uint64 i = 0; i < len; i++) {
        _out[0] += _Lin[i] * _Rin[i];
      }
    }
    void Vectordot_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &Lin,
                                const boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len, const bool &is_conj) {
      cytnx_int16 *_out = (cytnx_int16 *)out->data();
      cytnx_int16 *_Lin = (cytnx_int16 *)Lin->data();
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->data();

      _out[0] = 0;
      for (cytnx_uint64 i = 0; i < len; i++) {
        _out[0] += _Lin[i] * _Rin[i];
      }
    }
    void Vectordot_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &Lin,
                                const boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len, const bool &is_conj) {
      cytnx_uint16 *_out = (cytnx_uint16 *)out->data();
      cytnx_uint16 *_Lin = (cytnx_uint16 *)Lin->data();
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->data();

      _out[0] = 0;
      for (cytnx_uint64 i = 0; i < len; i++) {
        _out[0] += _Lin[i] * _Rin[i];
      }
    }
    void Vectordot_internal_b(boost::intrusive_ptr<Storage_base> &out,
                              const boost::intrusive_ptr<Storage_base> &Lin,
                              const boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const bool &is_conj) {
      cytnx_error_msg(1,
                      "[ERROR][Vectordot_internal_b][FATAL Invalid internal call.] No internal "
                      "function for vectordot of bool type.%s",
                      "\n");
    }

  }  // namespace linalg_internal
}  // namespace cytnx
