#include "utils/utils_internal_cpu/Cast_cpu.hpp"
#include "Storage.hpp"
#ifdef UNI_OMP
  #include <omp.h>
#endif

using namespace std;

namespace cytnx {
  namespace utils_internal {

    void Cast_cpu_cdtcd(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
        out->Init(len_in);
      }
      memcpy(out->Mem, in->Mem, sizeof(cytnx_complex128) * len_in);
    }

    void Cast_cpu_cdtcf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
        out->Init(len_in);
      }

      cytnx_complex128* _in = static_cast<cytnx_complex128*>(in->Mem);
      cytnx_complex64* _out = static_cast<cytnx_complex64*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }

    void Cast_cpu_cftcd(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
        out->Init(len_in);
      }
      cytnx_complex64* _in = static_cast<cytnx_complex64*>(in->Mem);
      cytnx_complex128* _out = static_cast<cytnx_complex128*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }

    void Cast_cpu_cftcf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
        out->Init(len_in);
      }
      memcpy(out->Mem, in->Mem, sizeof(cytnx_complex64) * len_in);
    }

    void Cast_cpu_dtcd(const boost::intrusive_ptr<Storage_base>& in,
                       boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                       const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
        out->Init(len_in);
      }
      cytnx_double* _in = static_cast<cytnx_double*>(in->Mem);
      cytnx_complex128* _out = static_cast<cytnx_complex128*>(out->Mem);

      memset(_out, 0, sizeof(cytnx_complex128) * len_in);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }

    void Cast_cpu_dtcf(const boost::intrusive_ptr<Storage_base>& in,
                       boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                       const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
        out->Init(len_in);
      }
      cytnx_double* _in = static_cast<cytnx_double*>(in->Mem);
      cytnx_complex64* _out = static_cast<cytnx_complex64*>(out->Mem);
      memset(_out, 0, sizeof(cytnx_complex64) * len_in);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }

    void Cast_cpu_dtd(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
        out->Init(len_in);
      }
      memcpy(out->Mem, in->Mem, sizeof(cytnx_double) * len_in);
    }
    void Cast_cpu_dtf(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
        out->Init(len_in);
      }
      cytnx_double* _in = static_cast<cytnx_double*>(in->Mem);
      cytnx_float* _out = static_cast<cytnx_float*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_dti64(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
        out->Init(len_in);
      }
      cytnx_double* _in = static_cast<cytnx_double*>(in->Mem);
      cytnx_int64* _out = static_cast<cytnx_int64*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_dtu64(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
        out->Init(len_in);
      }
      cytnx_double* _in = static_cast<cytnx_double*>(in->Mem);
      cytnx_uint64* _out = static_cast<cytnx_uint64*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_dti32(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
        out->Init(len_in);
      }
      cytnx_double* _in = static_cast<cytnx_double*>(in->Mem);
      cytnx_int32* _out = static_cast<cytnx_int32*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_dtu32(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
        out->Init(len_in);
      }
      cytnx_double* _in = static_cast<cytnx_double*>(in->Mem);
      cytnx_uint32* _out = static_cast<cytnx_uint32*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_dti16(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int16Storage());
        out->Init(len_in);
      }
      cytnx_double* _in = static_cast<cytnx_double*>(in->Mem);
      cytnx_int16* _out = static_cast<cytnx_int16*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_dtu16(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint16Storage());
        out->Init(len_in);
      }
      cytnx_double* _in = static_cast<cytnx_double*>(in->Mem);
      cytnx_uint16* _out = static_cast<cytnx_uint16*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_dtb(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new BoolStorage());
        out->Init(len_in);
      }
      cytnx_double* _in = static_cast<cytnx_double*>(in->Mem);
      cytnx_bool* _out = static_cast<cytnx_bool*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = bool(_in[i]);
      }
    }
    //------------------------------
    void Cast_cpu_ftcd(const boost::intrusive_ptr<Storage_base>& in,
                       boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                       const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
        out->Init(len_in);
      }
      cytnx_float* _in = static_cast<cytnx_float*>(in->Mem);
      cytnx_complex128* _out = static_cast<cytnx_complex128*>(out->Mem);
      memset(_out, 0, sizeof(cytnx_complex128) * len_in);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }
    void Cast_cpu_ftcf(const boost::intrusive_ptr<Storage_base>& in,
                       boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                       const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
        out->Init(len_in);
      }
      cytnx_float* _in = static_cast<cytnx_float*>(in->Mem);
      cytnx_complex64* _out = static_cast<cytnx_complex64*>(out->Mem);
      memset(_out, 0, sizeof(cytnx_complex64) * len_in);
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }
    void Cast_cpu_ftd(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
        out->Init(len_in);
      }
      cytnx_float* _in = static_cast<cytnx_float*>(in->Mem);
      cytnx_double* _out = static_cast<cytnx_double*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_ftf(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
        out->Init(len_in);
      }
      memcpy(out->Mem, in->Mem, sizeof(cytnx_float) * len_in);
    }
    void Cast_cpu_fti64(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
        out->Init(len_in);
      }
      cytnx_float* _in = static_cast<cytnx_float*>(in->Mem);
      cytnx_int64* _out = static_cast<cytnx_int64*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_ftu64(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
        out->Init(len_in);
      }
      cytnx_float* _in = static_cast<cytnx_float*>(in->Mem);
      cytnx_uint64* _out = static_cast<cytnx_uint64*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_fti32(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
        out->Init(len_in);
      }
      cytnx_float* _in = static_cast<cytnx_float*>(in->Mem);
      cytnx_int32* _out = static_cast<cytnx_int32*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_ftu32(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
        out->Init(len_in);
      }
      cytnx_float* _in = static_cast<cytnx_float*>(in->Mem);
      cytnx_uint32* _out = static_cast<cytnx_uint32*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_fti16(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int16Storage());
        out->Init(len_in);
      }
      cytnx_float* _in = static_cast<cytnx_float*>(in->Mem);
      cytnx_int16* _out = static_cast<cytnx_int16*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_ftu16(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint16Storage());
        out->Init(len_in);
      }
      cytnx_float* _in = static_cast<cytnx_float*>(in->Mem);
      cytnx_uint16* _out = static_cast<cytnx_uint16*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_ftb(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new BoolStorage());
        out->Init(len_in);
      }
      cytnx_float* _in = static_cast<cytnx_float*>(in->Mem);
      cytnx_bool* _out = static_cast<cytnx_bool*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = bool(_in[i]);
      }
    }
    //----------------------------
    void Cast_cpu_i64tcd(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
        out->Init(len_in);
      }
      cytnx_int64* _in = static_cast<cytnx_int64*>(in->Mem);
      cytnx_complex128* _out = static_cast<cytnx_complex128*>(out->Mem);
      memset(_out, 0, sizeof(cytnx_complex128) * len_in);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }
    void Cast_cpu_i64tcf(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
        out->Init(len_in);
      }
      cytnx_int64* _in = static_cast<cytnx_int64*>(in->Mem);
      cytnx_complex64* _out = static_cast<cytnx_complex64*>(out->Mem);
      memset(_out, 0, sizeof(cytnx_complex64) * len_in);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }
    void Cast_cpu_i64td(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
        out->Init(len_in);
      }
      cytnx_int64* _in = static_cast<cytnx_int64*>(in->Mem);
      cytnx_double* _out = static_cast<cytnx_double*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i64tf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
        out->Init(len_in);
      }
      cytnx_int64* _in = static_cast<cytnx_int64*>(in->Mem);
      cytnx_float* _out = static_cast<cytnx_float*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i64ti64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
        out->Init(len_in);
      }
      memcpy(out->Mem, in->Mem, sizeof(cytnx_int64) * len_in);
    }
    void Cast_cpu_i64tu64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
        out->Init(len_in);
      }
      cytnx_int64* _in = static_cast<cytnx_int64*>(in->Mem);
      cytnx_uint64* _out = static_cast<cytnx_uint64*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i64ti32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
        out->Init(len_in);
      }
      cytnx_int64* _in = static_cast<cytnx_int64*>(in->Mem);
      cytnx_int32* _out = static_cast<cytnx_int32*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i64tu32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
        out->Init(len_in);
      }
      cytnx_int64* _in = static_cast<cytnx_int64*>(in->Mem);
      cytnx_uint32* _out = static_cast<cytnx_uint32*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i64ti16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int16Storage());
        out->Init(len_in);
      }
      cytnx_int64* _in = static_cast<cytnx_int64*>(in->Mem);
      cytnx_int16* _out = static_cast<cytnx_int16*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i64tu16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint16Storage());
        out->Init(len_in);
      }
      cytnx_int64* _in = static_cast<cytnx_int64*>(in->Mem);
      cytnx_uint16* _out = static_cast<cytnx_uint16*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i64tb(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new BoolStorage());
        out->Init(len_in);
      }
      cytnx_int64* _in = static_cast<cytnx_int64*>(in->Mem);
      cytnx_bool* _out = static_cast<cytnx_bool*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = bool(_in[i]);
      }
    }

    //--------------------------------
    void Cast_cpu_u64tcd(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
        out->Init(len_in);
      }
      cytnx_uint64* _in = static_cast<cytnx_uint64*>(in->Mem);
      cytnx_complex128* _out = static_cast<cytnx_complex128*>(out->Mem);
      memset(_out, 0, sizeof(cytnx_complex128) * len_in);
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }
    void Cast_cpu_u64tcf(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
        out->Init(len_in);
      }
      cytnx_uint64* _in = static_cast<cytnx_uint64*>(in->Mem);
      cytnx_complex64* _out = static_cast<cytnx_complex64*>(out->Mem);
      memset(_out, 0, sizeof(cytnx_complex64) * len_in);
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }
    void Cast_cpu_u64td(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
        out->Init(len_in);
      }
      cytnx_uint64* _in = static_cast<cytnx_uint64*>(in->Mem);
      cytnx_double* _out = static_cast<cytnx_double*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u64tf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
        out->Init(len_in);
      }
      cytnx_uint64* _in = static_cast<cytnx_uint64*>(in->Mem);
      cytnx_float* _out = static_cast<cytnx_float*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u64ti64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
        out->Init(len_in);
      }
      cytnx_uint64* _in = static_cast<cytnx_uint64*>(in->Mem);
      cytnx_int64* _out = static_cast<cytnx_int64*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u64tu64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
        out->Init(len_in);
      }
      memcpy(out->Mem, in->Mem, sizeof(cytnx_uint64) * len_in);
    }
    void Cast_cpu_u64ti32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
        out->Init(len_in);
      }
      cytnx_uint64* _in = static_cast<cytnx_uint64*>(in->Mem);
      cytnx_int32* _out = static_cast<cytnx_int32*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u64tu32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
        out->Init(len_in);
      }
      cytnx_uint64* _in = static_cast<cytnx_uint64*>(in->Mem);
      cytnx_uint32* _out = static_cast<cytnx_uint32*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u64ti16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int16Storage());
        out->Init(len_in);
      }
      cytnx_uint64* _in = static_cast<cytnx_uint64*>(in->Mem);
      cytnx_int16* _out = static_cast<cytnx_int16*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u64tu16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint16Storage());
        out->Init(len_in);
      }
      cytnx_uint64* _in = static_cast<cytnx_uint64*>(in->Mem);
      cytnx_uint16* _out = static_cast<cytnx_uint16*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u64tb(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new BoolStorage());
        out->Init(len_in);
      }
      cytnx_uint64* _in = static_cast<cytnx_uint64*>(in->Mem);
      cytnx_bool* _out = static_cast<cytnx_bool*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = bool(_in[i]);
      }
    }

    //------------------------------------------------
    void Cast_cpu_i32tcd(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
        out->Init(len_in);
      }
      cytnx_int32* _in = static_cast<cytnx_int32*>(in->Mem);
      cytnx_complex128* _out = static_cast<cytnx_complex128*>(out->Mem);
      memset(_out, 0, sizeof(cytnx_complex128) * len_in);
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }
    void Cast_cpu_i32tcf(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
        out->Init(len_in);
      }
      cytnx_int32* _in = static_cast<cytnx_int32*>(in->Mem);
      cytnx_complex64* _out = static_cast<cytnx_complex64*>(out->Mem);
      memset(_out, 0, sizeof(cytnx_complex64) * len_in);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }
    void Cast_cpu_i32td(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
        out->Init(len_in);
      }
      cytnx_int32* _in = static_cast<cytnx_int32*>(in->Mem);
      cytnx_double* _out = static_cast<cytnx_double*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i32tf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
        out->Init(len_in);
      }
      cytnx_int32* _in = static_cast<cytnx_int32*>(in->Mem);
      cytnx_float* _out = static_cast<cytnx_float*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i32ti64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
        out->Init(len_in);
      }
      cytnx_int32* _in = static_cast<cytnx_int32*>(in->Mem);
      cytnx_int64* _out = static_cast<cytnx_int64*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i32tu64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
        out->Init(len_in);
      }
      cytnx_int32* _in = static_cast<cytnx_int32*>(in->Mem);
      cytnx_uint64* _out = static_cast<cytnx_uint64*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i32ti32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
        out->Init(len_in);
      }
      memcpy(out->Mem, in->Mem, sizeof(cytnx_int32) * len_in);
    }
    void Cast_cpu_i32tu32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
        out->Init(len_in);
      }
      cytnx_int32* _in = static_cast<cytnx_int32*>(in->Mem);
      cytnx_uint32* _out = static_cast<cytnx_uint32*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i32tu16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint16Storage());
        out->Init(len_in);
      }
      cytnx_int32* _in = static_cast<cytnx_int32*>(in->Mem);
      cytnx_uint16* _out = static_cast<cytnx_uint16*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i32ti16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int16Storage());
        out->Init(len_in);
      }
      cytnx_int32* _in = static_cast<cytnx_int32*>(in->Mem);
      cytnx_int16* _out = static_cast<cytnx_int16*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i32tb(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new BoolStorage());
        out->Init(len_in);
      }
      cytnx_int32* _in = static_cast<cytnx_int32*>(in->Mem);
      cytnx_bool* _out = static_cast<cytnx_bool*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = bool(_in[i]);
      }
    }
    //-----------------------------------
    void Cast_cpu_u32tcd(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
        out->Init(len_in);
      }
      cytnx_uint32* _in = static_cast<cytnx_uint32*>(in->Mem);
      cytnx_complex128* _out = static_cast<cytnx_complex128*>(out->Mem);
      memset(_out, 0, sizeof(cytnx_complex128) * len_in);
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }
    void Cast_cpu_u32tcf(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
        out->Init(len_in);
      }
      cytnx_uint32* _in = static_cast<cytnx_uint32*>(in->Mem);
      cytnx_complex64* _out = static_cast<cytnx_complex64*>(out->Mem);
      memset(_out, 0, sizeof(cytnx_complex64) * len_in);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }
    void Cast_cpu_u32td(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
        out->Init(len_in);
      }
      cytnx_uint32* _in = static_cast<cytnx_uint32*>(in->Mem);
      cytnx_double* _out = static_cast<cytnx_double*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u32tf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
        out->Init(len_in);
      }
      cytnx_uint32* _in = static_cast<cytnx_uint32*>(in->Mem);
      cytnx_float* _out = static_cast<cytnx_float*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u32ti64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
        out->Init(len_in);
      }
      cytnx_uint32* _in = static_cast<cytnx_uint32*>(in->Mem);
      cytnx_int64* _out = static_cast<cytnx_int64*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u32tu64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
        out->Init(len_in);
      }
      cytnx_uint32* _in = static_cast<cytnx_uint32*>(in->Mem);
      cytnx_uint64* _out = static_cast<cytnx_uint64*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u32ti32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
        out->Init(len_in);
      }
      cytnx_uint32* _in = static_cast<cytnx_uint32*>(in->Mem);
      cytnx_int32* _out = static_cast<cytnx_int32*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u32tu32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
        out->Init(len_in);
      }
      memcpy(out->Mem, in->Mem, sizeof(cytnx_uint32) * len_in);
    }
    void Cast_cpu_u32tu16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint16Storage());
        out->Init(len_in);
      }
      cytnx_uint32* _in = static_cast<cytnx_uint32*>(in->Mem);
      cytnx_uint16* _out = static_cast<cytnx_uint16*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u32ti16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int16Storage());
        out->Init(len_in);
      }
      cytnx_uint32* _in = static_cast<cytnx_uint32*>(in->Mem);
      cytnx_int16* _out = static_cast<cytnx_int16*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u32tb(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new BoolStorage());
        out->Init(len_in);
      }
      cytnx_uint32* _in = static_cast<cytnx_uint32*>(in->Mem);
      cytnx_bool* _out = static_cast<cytnx_bool*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = bool(_in[i]);
      }
    }
    //------------------------------
    void Cast_cpu_u16tcd(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
        out->Init(len_in);
      }
      cytnx_uint16* _in = static_cast<cytnx_uint16*>(in->Mem);
      cytnx_complex128* _out = static_cast<cytnx_complex128*>(out->Mem);
      memset(_out, 0, sizeof(cytnx_complex128) * len_in);
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }
    void Cast_cpu_u16tcf(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
        out->Init(len_in);
      }
      cytnx_uint16* _in = static_cast<cytnx_uint16*>(in->Mem);
      cytnx_complex64* _out = static_cast<cytnx_complex64*>(out->Mem);
      memset(_out, 0, sizeof(cytnx_complex64) * len_in);
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }
    void Cast_cpu_u16td(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
        out->Init(len_in);
      }
      cytnx_uint16* _in = static_cast<cytnx_uint16*>(in->Mem);
      cytnx_double* _out = static_cast<cytnx_double*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u16tf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
        out->Init(len_in);
      }
      cytnx_uint16* _in = static_cast<cytnx_uint16*>(in->Mem);
      cytnx_float* _out = static_cast<cytnx_float*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u16ti64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
        out->Init(len_in);
      }
      cytnx_uint16* _in = static_cast<cytnx_uint16*>(in->Mem);
      cytnx_int64* _out = static_cast<cytnx_int64*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u16tu64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
        out->Init(len_in);
      }
      cytnx_uint16* _in = static_cast<cytnx_uint16*>(in->Mem);
      cytnx_uint64* _out = static_cast<cytnx_uint64*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u16ti32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
        out->Init(len_in);
      }
      cytnx_uint16* _in = static_cast<cytnx_uint16*>(in->Mem);
      cytnx_int32* _out = static_cast<cytnx_int32*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u16tu32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
        out->Init(len_in);
      }
      cytnx_uint16* _in = static_cast<cytnx_uint16*>(in->Mem);
      cytnx_uint32* _out = static_cast<cytnx_uint32*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u16tu16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint16Storage());
        out->Init(len_in);
      }
      memcpy(out->Mem, in->Mem, sizeof(cytnx_uint16) * len_in);
    }
    void Cast_cpu_u16ti16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int16Storage());
        out->Init(len_in);
      }
      cytnx_uint16* _in = static_cast<cytnx_uint16*>(in->Mem);
      cytnx_int16* _out = static_cast<cytnx_int16*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_u16tb(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new BoolStorage());
        out->Init(len_in);
      }
      cytnx_uint16* _in = static_cast<cytnx_uint16*>(in->Mem);
      cytnx_bool* _out = static_cast<cytnx_bool*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = bool(_in[i]);
      }
    }
    //--------------------------------
    void Cast_cpu_i16tcd(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
        out->Init(len_in);
      }
      cytnx_int16* _in = static_cast<cytnx_int16*>(in->Mem);
      cytnx_complex128* _out = static_cast<cytnx_complex128*>(out->Mem);
      memset(_out, 0, sizeof(cytnx_complex128) * len_in);
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }
    void Cast_cpu_i16tcf(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
        out->Init(len_in);
      }
      cytnx_int16* _in = static_cast<cytnx_int16*>(in->Mem);
      cytnx_complex64* _out = static_cast<cytnx_complex64*>(out->Mem);
      memset(_out, 0, sizeof(cytnx_complex64) * len_in);
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }
    void Cast_cpu_i16td(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
        out->Init(len_in);
      }
      cytnx_int16* _in = static_cast<cytnx_int16*>(in->Mem);
      cytnx_double* _out = static_cast<cytnx_double*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i16tf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
        out->Init(len_in);
      }
      cytnx_int16* _in = static_cast<cytnx_int16*>(in->Mem);
      cytnx_float* _out = static_cast<cytnx_float*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i16ti64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
        out->Init(len_in);
      }
      cytnx_int16* _in = static_cast<cytnx_int16*>(in->Mem);
      cytnx_int64* _out = static_cast<cytnx_int64*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i16tu64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
        out->Init(len_in);
      }
      cytnx_int16* _in = static_cast<cytnx_int16*>(in->Mem);
      cytnx_uint64* _out = static_cast<cytnx_uint64*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i16ti32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
        out->Init(len_in);
      }
      cytnx_int16* _in = static_cast<cytnx_int16*>(in->Mem);
      cytnx_int32* _out = static_cast<cytnx_int32*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i16tu32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
        out->Init(len_in);
      }
      cytnx_int16* _in = static_cast<cytnx_int16*>(in->Mem);
      cytnx_uint32* _out = static_cast<cytnx_uint32*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i16tu16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint16Storage());
        out->Init(len_in);
      }
      cytnx_int16* _in = static_cast<cytnx_int16*>(in->Mem);
      cytnx_uint16* _out = static_cast<cytnx_uint16*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_i16ti16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int16Storage());
        out->Init(len_in);
      }
      memcpy(out->Mem, in->Mem, sizeof(cytnx_int16) * len_in);
    }
    void Cast_cpu_i16tb(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new BoolStorage());
        out->Init(len_in);
      }
      cytnx_int16* _in = static_cast<cytnx_int16*>(in->Mem);
      cytnx_bool* _out = static_cast<cytnx_bool*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = bool(_in[i]);
      }
    }
    //---------------------------
    void Cast_cpu_btcd(const boost::intrusive_ptr<Storage_base>& in,
                       boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                       const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
        out->Init(len_in);
      }
      cytnx_bool* _in = static_cast<cytnx_bool*>(in->Mem);
      cytnx_complex128* _out = static_cast<cytnx_complex128*>(out->Mem);
      memset(_out, 0, sizeof(cytnx_complex128) * len_in);
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }
    void Cast_cpu_btcf(const boost::intrusive_ptr<Storage_base>& in,
                       boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                       const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
        out->Init(len_in);
      }
      cytnx_bool* _in = static_cast<cytnx_bool*>(in->Mem);
      cytnx_complex64* _out = static_cast<cytnx_complex64*>(out->Mem);
      memset(_out, 0, sizeof(cytnx_complex64) * len_in);
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i].real(_in[i]);
      }
    }
    void Cast_cpu_btd(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
        out->Init(len_in);
      }
      cytnx_bool* _in = static_cast<cytnx_bool*>(in->Mem);
      cytnx_double* _out = static_cast<cytnx_double*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_btf(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
        out->Init(len_in);
      }
      cytnx_bool* _in = static_cast<cytnx_bool*>(in->Mem);
      cytnx_float* _out = static_cast<cytnx_float*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_bti64(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
        out->Init(len_in);
      }
      cytnx_bool* _in = static_cast<cytnx_bool*>(in->Mem);
      cytnx_int64* _out = static_cast<cytnx_int64*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_btu64(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
        out->Init(len_in);
      }
      cytnx_bool* _in = static_cast<cytnx_bool*>(in->Mem);
      cytnx_uint64* _out = static_cast<cytnx_uint64*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_bti32(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
        out->Init(len_in);
      }
      cytnx_bool* _in = static_cast<cytnx_bool*>(in->Mem);
      cytnx_int32* _out = static_cast<cytnx_int32*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_btu32(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
        out->Init(len_in);
      }
      cytnx_bool* _in = static_cast<cytnx_bool*>(in->Mem);
      cytnx_uint32* _out = static_cast<cytnx_uint32*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_btu16(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Uint16Storage());
        out->Init(len_in);
      }
      cytnx_bool* _in = static_cast<cytnx_bool*>(in->Mem);
      cytnx_uint16* _out = static_cast<cytnx_uint16*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_bti16(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new Int16Storage());
        out->Init(len_in);
      }
      cytnx_bool* _in = static_cast<cytnx_bool*>(in->Mem);
      cytnx_int16* _out = static_cast<cytnx_int16*>(out->Mem);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (unsigned long long i = 0; i < len_in; i++) {
        _out[i] = _in[i];
      }
    }
    void Cast_cpu_btb(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc) {
      if (is_alloc) {
        out = boost::intrusive_ptr<Storage_base>(new BoolStorage());
        out->Init(len_in);
      }
      memcpy(out->Mem, in->Mem, sizeof(cytnx_bool) * len_in);
    }

  }  // namespace utils_internal
}  // namespace cytnx
