#include "utils/utils_internal_cpu/Fill_cpu.hpp"
#include "Storage.hpp"
#ifdef UNI_OMP
  #include <omp.h>
#endif

using namespace std;

namespace cytnx {
  namespace utils_internal {

    void Fill_cpu_cd(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_complex128* ptr = (cytnx_complex128*)in;
      cytnx_complex128 _val = *((cytnx_complex128*)val);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 i = 0; i < Nelem; i++) {
        ptr[i] = _val;
      }
    }

    void Fill_cpu_cf(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_complex64* ptr = (cytnx_complex64*)in;
      cytnx_complex64 _val = *((cytnx_complex64*)val);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 i = 0; i < Nelem; i++) {
        ptr[i] = _val;
      }
    }

    void Fill_cpu_d(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_double* ptr = (cytnx_double*)in;
      cytnx_double _val = *((cytnx_double*)val);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 i = 0; i < Nelem; i++) {
        ptr[i] = _val;
      }
    }

    void Fill_cpu_f(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_float* ptr = (cytnx_float*)in;
      cytnx_float _val = *((cytnx_float*)val);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 i = 0; i < Nelem; i++) {
        ptr[i] = _val;
      }
    }

    void Fill_cpu_i64(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_int64* ptr = (cytnx_int64*)in;
      cytnx_int64 _val = *((cytnx_int64*)val);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 i = 0; i < Nelem; i++) {
        ptr[i] = _val;
      }
    }

    void Fill_cpu_u64(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_uint64* ptr = (cytnx_uint64*)in;
      cytnx_uint64 _val = *((cytnx_uint64*)val);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 i = 0; i < Nelem; i++) {
        ptr[i] = _val;
      }
    }

    void Fill_cpu_i32(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_int32* ptr = (cytnx_int32*)in;
      cytnx_int32 _val = *((cytnx_int32*)val);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 i = 0; i < Nelem; i++) {
        ptr[i] = _val;
      }
    }

    void Fill_cpu_u32(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_uint32* ptr = (cytnx_uint32*)in;
      cytnx_uint32 _val = *((cytnx_uint32*)val);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 i = 0; i < Nelem; i++) {
        ptr[i] = _val;
      }
    }

    void Fill_cpu_i16(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_int16* ptr = (cytnx_int16*)in;
      cytnx_int16 _val = *((cytnx_int16*)val);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 i = 0; i < Nelem; i++) {
        ptr[i] = _val;
      }
    }

    void Fill_cpu_u16(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_uint16* ptr = (cytnx_uint16*)in;
      cytnx_uint16 _val = *((cytnx_uint16*)val);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 i = 0; i < Nelem; i++) {
        ptr[i] = _val;
      }
    }

    void Fill_cpu_b(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_bool* ptr = (cytnx_bool*)in;
      cytnx_bool _val = *((cytnx_bool*)val);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 i = 0; i < Nelem; i++) {
        ptr[i] = _val;
      }
    }

  }  // namespace utils_internal
}  // namespace cytnx
