#include "Sum_internal.hpp"
#include "cytnx_error.hpp"
#include "lapack_wrapper.hpp"
#include <cfloat>
#include <limits>

#ifdef UNI_OMP
  #include <omp.h>
#endif
using namespace std;

namespace cytnx {
  namespace linalg_internal {

    void Sum_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                          const char &type) {
      cytnx_uint64 *_ten = (cytnx_uint64 *)ten->Mem;
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;

#ifdef UNI_OMP
      vector<cytnx_uint64> buf;
      unsigned int Nproc;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0) Nproc = omp_get_num_threads();
      }
      buf = vector<cytnx_uint64>(Nproc, 0);

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        buf[omp_get_thread_num()] += _ten[n];
      }

      for (int i = 1; i < Nproc; i++) {
        buf[0] += buf[i];
      }
      _out[0] = buf[0];

#else
      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
#endif
    }

    void Sum_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                          const char &type) {
      cytnx_int64 *_ten = (cytnx_int64 *)ten->Mem;
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;

#ifdef UNI_OMP
      vector<cytnx_int64> buf;
      unsigned int Nproc = 1;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0) Nproc = omp_get_num_threads();
      }
      buf = vector<cytnx_int64>(Nproc, 0);

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        buf[omp_get_thread_num()] += _ten[n];
      }

      for (int i = 1; i < Nproc; i++) {
        buf[0] += buf[i];
      }
      _out[0] = buf[0];

#else
      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
#endif
    }

    void Sum_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                          const char &type) {
      cytnx_uint32 *_ten = (cytnx_uint32 *)ten->Mem;
      cytnx_uint32 *_out = (cytnx_uint32 *)out->Mem;

#ifdef UNI_OMP
      vector<cytnx_uint32> buf;
      unsigned int Nproc;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0) Nproc = omp_get_num_threads();
      }
      buf = vector<cytnx_uint32>(Nproc, 0);

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        buf[omp_get_thread_num()] += _ten[n];
      }

      for (int i = 1; i < Nproc; i++) {
        buf[0] += buf[i];
      }
      _out[0] = buf[0];

#else
      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
#endif
    }

    void Sum_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                          const char &type) {
      cytnx_int32 *_ten = (cytnx_int32 *)ten->Mem;
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;

#ifdef UNI_OMP
      vector<cytnx_int32> buf;
      unsigned int Nproc;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0) Nproc = omp_get_num_threads();
      }
      buf = vector<cytnx_int32>(Nproc, 0);

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        buf[omp_get_thread_num()] += _ten[n];
      }

      for (int i = 1; i < Nproc; i++) {
        buf[0] += buf[i];
      }
      _out[0] = buf[0];

#else
      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
#endif
    }

    void Sum_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                          const char &type) {
      cytnx_uint16 *_ten = (cytnx_uint16 *)ten->Mem;
      cytnx_uint16 *_out = (cytnx_uint16 *)out->Mem;

#ifdef UNI_OMP
      vector<cytnx_uint16> buf;
      unsigned int Nproc;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0) Nproc = omp_get_num_threads();
      }
      buf = vector<cytnx_uint16>(Nproc, 0);

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        buf[omp_get_thread_num()] += _ten[n];
      }

      for (int i = 1; i < Nproc; i++) {
        buf[0] += buf[i];
      }
      _out[0] = buf[0];

#else
      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
#endif
    }

    void Sum_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                          const char &type) {
      cytnx_int16 *_ten = (cytnx_int16 *)ten->Mem;
      cytnx_int16 *_out = (cytnx_int16 *)out->Mem;

#ifdef UNI_OMP
      vector<cytnx_int16> buf;
      unsigned int Nproc;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0) Nproc = omp_get_num_threads();
      }
      buf = vector<cytnx_int16>(Nproc, 0);

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        buf[omp_get_thread_num()] += _ten[n];
      }

      for (int i = 1; i < Nproc; i++) {
        buf[0] += buf[i];
      }
      _out[0] = buf[0];

#else
      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
#endif
    }

    void Sum_internal_d(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                        const char &type) {
      cytnx_double *_ten = (cytnx_double *)ten->Mem;
      cytnx_double *_out = (cytnx_double *)out->Mem;

#ifdef UNI_OMP
      vector<cytnx_double> buf;
      unsigned int Nproc;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0) Nproc = omp_get_num_threads();
      }
      buf = vector<cytnx_double>(Nproc, 0);

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        buf[omp_get_thread_num()] += _ten[n];
      }

      for (int i = 1; i < Nproc; i++) {
        buf[0] += buf[i];
      }
      _out[0] = buf[0];

#else
      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
#endif
    }

    void Sum_internal_f(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                        const char &type) {
      cytnx_float *_ten = (cytnx_float *)ten->Mem;
      cytnx_float *_out = (cytnx_float *)out->Mem;

#ifdef UNI_OMP
      vector<cytnx_float> buf;
      unsigned int Nproc;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0) Nproc = omp_get_num_threads();
      }
      buf = vector<cytnx_float>(Nproc, 0);

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        buf[omp_get_thread_num()] += _ten[n];
      }

      for (int i = 1; i < Nproc; i++) {
        buf[0] += buf[i];
      }
      _out[0] = buf[0];

#else
      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
#endif
    }
    void Sum_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                         const char &type) {
      cytnx_complex128 *_ten = (cytnx_complex128 *)ten->Mem;
      cytnx_complex128 *_out = (cytnx_complex128 *)out->Mem;

#ifdef UNI_OMP
      vector<cytnx_complex128> buf;
      unsigned int Nproc;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0) Nproc = omp_get_num_threads();
      }
      buf = vector<cytnx_complex128>(Nproc, 0);

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        buf[omp_get_thread_num()] += _ten[n];
      }

      for (int i = 1; i < Nproc; i++) {
        buf[0] += buf[i];
      }
      _out[0] = buf[0];

#else
      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
#endif
    }

    void Sum_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                         const char &type) {
      cytnx_complex64 *_ten = (cytnx_complex64 *)ten->Mem;
      cytnx_complex64 *_out = (cytnx_complex64 *)out->Mem;

#ifdef UNI_OMP
      vector<cytnx_complex64> buf;
      unsigned int Nproc;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0) Nproc = omp_get_num_threads();
      }
      buf = vector<cytnx_complex64>(Nproc, 0);

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        buf[omp_get_thread_num()] += _ten[n];
      }

      for (int i = 1; i < Nproc; i++) {
        buf[0] += buf[i];
      }
      _out[0] = buf[0];

#else
      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
#endif
    }

    void Sum_internal_b(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                        const char &type) {
      cytnx_error_msg(true, "[ERROR][Sum_internal] bool does not have Sum support.%s", "\n");
    }

  }  // namespace linalg_internal

}  // namespace cytnx
