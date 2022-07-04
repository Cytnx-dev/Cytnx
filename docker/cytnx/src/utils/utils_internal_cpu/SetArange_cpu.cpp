#include "utils/utils_internal_cpu/SetArange_cpu.hpp"
using namespace std;

namespace cytnx {
  namespace utils_internal {
    void SetArange_cpu_cd(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                          const cytnx_double &end, const cytnx_double &step,
                          const cytnx_uint64 &Nelem) {
      cytnx_complex128 *ptr = (cytnx_complex128 *)in->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        ptr[n].real(start + n * step);
        ptr[n].imag(0);
      }
    }
    void SetArange_cpu_cf(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                          const cytnx_double &end, const cytnx_double &step,
                          const cytnx_uint64 &Nelem) {
      cytnx_complex64 *ptr = (cytnx_complex64 *)in->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        ptr[n].real(start + n * step);
        ptr[n].imag(0);
      }
    }
    void SetArange_cpu_d(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                         const cytnx_double &end, const cytnx_double &step,
                         const cytnx_uint64 &Nelem) {
      cytnx_double *ptr = (cytnx_double *)in->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        ptr[n] = start + n * step;
      }
    }
    void SetArange_cpu_f(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                         const cytnx_double &end, const cytnx_double &step,
                         const cytnx_uint64 &Nelem) {
      cytnx_float *ptr = (cytnx_float *)in->Mem;
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        ptr[n] = start + n * step;
      }
    }
    void SetArange_cpu_i64(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem) {
      cytnx_int64 *ptr = (cytnx_int64 *)in->Mem;
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        ptr[n] = start + n * step;
      }
    }
    void SetArange_cpu_u64(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem) {
      cytnx_uint64 *ptr = (cytnx_uint64 *)in->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        ptr[n] = start + n * step;
      }
    }
    void SetArange_cpu_i32(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem) {
      cytnx_int32 *ptr = (cytnx_int32 *)in->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        ptr[n] = start + n * step;
      }
    }
    void SetArange_cpu_u32(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem) {
      cytnx_uint32 *ptr = (cytnx_uint32 *)in->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        ptr[n] = start + n * step;
      }
    }
    void SetArange_cpu_i16(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem) {
      cytnx_int16 *ptr = (cytnx_int16 *)in->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        ptr[n] = start + n * step;
      }
    }
    void SetArange_cpu_u16(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem) {
      cytnx_uint16 *ptr = (cytnx_uint16 *)in->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        ptr[n] = start + n * step;
      }
    }
    void SetArange_cpu_b(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                         const cytnx_double &end, const cytnx_double &step,
                         const cytnx_uint64 &Nelem) {
      cytnx_bool *ptr = (cytnx_bool *)in->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        ptr[n] = start + n * step;
      }
    }
  }  // namespace utils_internal
}  // namespace cytnx
