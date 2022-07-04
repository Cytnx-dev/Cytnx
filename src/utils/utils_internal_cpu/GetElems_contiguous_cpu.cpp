#include "utils/utils_internal_cpu/GetElems_contiguous_cpu.hpp"

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {
  namespace utils_internal {

    void GetElems_contiguous_cpu_cd(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nelem_grp) {
      // Start copy elem:
      cytnx_complex128 *elem_ptr_ = static_cast<cytnx_complex128 *>(in);
      cytnx_complex128 *new_elem_ptr_ = static_cast<cytnx_complex128 *>(out);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < TotalElem; n++) {
        // map from mem loc of new tensor to old tensor
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = n;
        for (cytnx_uint32 r = 0; r < offj.size(); r++) {
          if (locators[r].size())
            Loc += locators[r][tmpn / new_offj[r]] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
        }
        memcpy(new_elem_ptr_ + n * Nelem_grp, elem_ptr_ + Loc * Nelem_grp,
               sizeof(cytnx_complex128) * Nelem_grp);
        // new_elem_ptr_[n] = elem_ptr_[Loc];
      }
    }
    void GetElems_contiguous_cpu_cf(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nelem_grp) {
      // Start copy elem:
      cytnx_complex64 *elem_ptr_ = static_cast<cytnx_complex64 *>(in);
      cytnx_complex64 *new_elem_ptr_ = static_cast<cytnx_complex64 *>(out);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < TotalElem; n++) {
        // map from mem loc of new tensor to old tensor
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = n;
        for (cytnx_uint32 r = 0; r < offj.size(); r++) {
          if (locators[r].size())
            Loc += locators[r][tmpn / new_offj[r]] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
        }
        memcpy(new_elem_ptr_ + n * Nelem_grp, elem_ptr_ + Loc * Nelem_grp,
               sizeof(cytnx_complex64) * Nelem_grp);
        // new_elem_ptr_[n] = elem_ptr_[Loc];
      }
    }
    void GetElems_contiguous_cpu_d(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nelem_grp) {
      // Start copy elem:
      cytnx_double *elem_ptr_ = static_cast<cytnx_double *>(in);
      cytnx_double *new_elem_ptr_ = static_cast<cytnx_double *>(out);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < TotalElem; n++) {
        // map from mem loc of new tensor to old tensor
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = n;
        for (cytnx_uint32 r = 0; r < offj.size(); r++) {
          if (locators[r].size())
            Loc += locators[r][tmpn / new_offj[r]] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
        }
        // std::cout << n << " " << Loc << std::endl;
        memcpy(new_elem_ptr_ + n * Nelem_grp, elem_ptr_ + Loc * Nelem_grp,
               sizeof(cytnx_double) * Nelem_grp);
        // new_elem_ptr_[n] = elem_ptr_[Loc];
      }
    }
    void GetElems_contiguous_cpu_f(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nelem_grp) {
      // Start copy elem:
      cytnx_float *elem_ptr_ = static_cast<cytnx_float *>(in);
      cytnx_float *new_elem_ptr_ = static_cast<cytnx_float *>(out);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < TotalElem; n++) {
        // map from mem loc of new tensor to old tensor
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = n;
        for (cytnx_uint32 r = 0; r < offj.size(); r++) {
          if (locators[r].size())
            Loc += locators[r][tmpn / new_offj[r]] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
        }
        memcpy(new_elem_ptr_ + n * Nelem_grp, elem_ptr_ + Loc * Nelem_grp,
               sizeof(cytnx_float) * Nelem_grp);
        // new_elem_ptr_[n] = elem_ptr_[Loc];
      }
    }
    void GetElems_contiguous_cpu_i64(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                                     const std::vector<cytnx_uint64> &new_offj,
                                     const std::vector<std::vector<cytnx_uint64>> &locators,
                                     const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nelem_grp) {
      // Start copy elem:
      cytnx_int64 *elem_ptr_ = static_cast<cytnx_int64 *>(in);
      cytnx_int64 *new_elem_ptr_ = static_cast<cytnx_int64 *>(out);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < TotalElem; n++) {
        // map from mem loc of new tensor to old tensor
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = n;
        for (cytnx_uint32 r = 0; r < offj.size(); r++) {
          if (locators[r].size())
            Loc += locators[r][tmpn / new_offj[r]] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
        }
        memcpy(new_elem_ptr_ + n * Nelem_grp, elem_ptr_ + Loc * Nelem_grp,
               sizeof(cytnx_int64) * Nelem_grp);
        // new_elem_ptr_[n] = elem_ptr_[Loc];
      }
    }
    void GetElems_contiguous_cpu_u64(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                                     const std::vector<cytnx_uint64> &new_offj,
                                     const std::vector<std::vector<cytnx_uint64>> &locators,
                                     const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nelem_grp) {
      // Start copy elem:
      cytnx_uint64 *elem_ptr_ = static_cast<cytnx_uint64 *>(in);
      cytnx_uint64 *new_elem_ptr_ = static_cast<cytnx_uint64 *>(out);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < TotalElem; n++) {
        // map from mem loc of new tensor to old tensor
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = n;
        for (cytnx_uint32 r = 0; r < offj.size(); r++) {
          if (locators[r].size())
            Loc += locators[r][tmpn / new_offj[r]] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
        }
        memcpy(new_elem_ptr_ + n * Nelem_grp, elem_ptr_ + Loc * Nelem_grp,
               sizeof(cytnx_uint64) * Nelem_grp);
        // new_elem_ptr_[n] = elem_ptr_[Loc];
      }
    }
    void GetElems_contiguous_cpu_i32(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                                     const std::vector<cytnx_uint64> &new_offj,
                                     const std::vector<std::vector<cytnx_uint64>> &locators,
                                     const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nelem_grp) {
      // Start copy elem:
      cytnx_int32 *elem_ptr_ = static_cast<cytnx_int32 *>(in);
      cytnx_int32 *new_elem_ptr_ = static_cast<cytnx_int32 *>(out);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < TotalElem; n++) {
        // map from mem loc of new tensor to old tensor
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = n;
        for (cytnx_uint32 r = 0; r < offj.size(); r++) {
          if (locators[r].size())
            Loc += locators[r][tmpn / new_offj[r]] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
        }
        memcpy(new_elem_ptr_ + n * Nelem_grp, elem_ptr_ + Loc * Nelem_grp,
               sizeof(cytnx_int32) * Nelem_grp);
        // new_elem_ptr_[n] = elem_ptr_[Loc];
      }
    }
    void GetElems_contiguous_cpu_u32(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                                     const std::vector<cytnx_uint64> &new_offj,
                                     const std::vector<std::vector<cytnx_uint64>> &locators,
                                     const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nelem_grp) {
      // Start copy elem:
      cytnx_uint32 *elem_ptr_ = static_cast<cytnx_uint32 *>(in);
      cytnx_uint32 *new_elem_ptr_ = static_cast<cytnx_uint32 *>(out);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < TotalElem; n++) {
        // map from mem loc of new tensor to old tensor
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = n;
        for (cytnx_uint32 r = 0; r < offj.size(); r++) {
          if (locators[r].size())
            Loc += locators[r][tmpn / new_offj[r]] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
        }
        memcpy(new_elem_ptr_ + n * Nelem_grp, elem_ptr_ + Loc * Nelem_grp,
               sizeof(cytnx_uint32) * Nelem_grp);
        // new_elem_ptr_[n] = elem_ptr_[Loc];
      }
    }
    void GetElems_contiguous_cpu_i16(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                                     const std::vector<cytnx_uint64> &new_offj,
                                     const std::vector<std::vector<cytnx_uint64>> &locators,
                                     const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nelem_grp) {
      // Start copy elem:
      cytnx_int16 *elem_ptr_ = static_cast<cytnx_int16 *>(in);
      cytnx_int16 *new_elem_ptr_ = static_cast<cytnx_int16 *>(out);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < TotalElem; n++) {
        // map from mem loc of new tensor to old tensor
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = n;
        for (cytnx_uint32 r = 0; r < offj.size(); r++) {
          if (locators[r].size())
            Loc += locators[r][tmpn / new_offj[r]] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
        }
        memcpy(new_elem_ptr_ + n * Nelem_grp, elem_ptr_ + Loc * Nelem_grp,
               sizeof(cytnx_int16) * Nelem_grp);
        // new_elem_ptr_[n] = elem_ptr_[Loc];
      }
    }
    void GetElems_contiguous_cpu_u16(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                                     const std::vector<cytnx_uint64> &new_offj,
                                     const std::vector<std::vector<cytnx_uint64>> &locators,
                                     const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nelem_grp) {
      // Start copy elem:
      cytnx_uint16 *elem_ptr_ = static_cast<cytnx_uint16 *>(in);
      cytnx_uint16 *new_elem_ptr_ = static_cast<cytnx_uint16 *>(out);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < TotalElem; n++) {
        // map from mem loc of new tensor to old tensor
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = n;
        for (cytnx_uint32 r = 0; r < offj.size(); r++) {
          if (locators[r].size())
            Loc += locators[r][tmpn / new_offj[r]] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
        }
        memcpy(new_elem_ptr_ + n * Nelem_grp, elem_ptr_ + Loc * Nelem_grp,
               sizeof(cytnx_uint16) * Nelem_grp);
        // new_elem_ptr_[n] = elem_ptr_[Loc];
      }
    }
    void GetElems_contiguous_cpu_b(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nelem_grp) {
      // Start copy elem:
      cytnx_bool *elem_ptr_ = static_cast<cytnx_bool *>(in);
      cytnx_bool *new_elem_ptr_ = static_cast<cytnx_bool *>(out);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < TotalElem; n++) {
        // map from mem loc of new tensor to old tensor
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = n;
        for (cytnx_uint32 r = 0; r < offj.size(); r++) {
          if (locators[r].size())
            Loc += locators[r][tmpn / new_offj[r]] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
        }
        memcpy(new_elem_ptr_ + n * Nelem_grp, elem_ptr_ + Loc * Nelem_grp,
               sizeof(cytnx_bool) * Nelem_grp);
        // new_elem_ptr_[n] = elem_ptr_[Loc];
      }
    }
  }  // namespace utils_internal
}  // namespace cytnx
