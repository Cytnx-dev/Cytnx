#include "SetElems_contiguous_cpu.hpp"
#include "utils/utils_internal_interface.hpp"
#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {
  namespace utils_internal {

    template <class T1>
    void SetElems_conti_cpu_impl_sametype(void *in, void *out,
                                          const std::vector<cytnx_uint64> &offj,
                                          const std::vector<cytnx_uint64> &new_offj,
                                          const std::vector<std::vector<cytnx_uint64>> &locators,
                                          const cytnx_uint64 &TotalElem,
                                          const cytnx_uint64 &Nunit) {
      // Start copy elem:
      T1 *new_elem_ptr_ = static_cast<T1 *>(in);
      T1 *elem_ptr_ = static_cast<T1 *>(out);

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
        memcpy(elem_ptr_ + Loc * Nunit, new_elem_ptr_ + n * Nunit, sizeof(T1) * Nunit);
        // for(cytnx_uint64 x=0;x<Nunit;x++)
        //     elem_ptr_[Loc*Nunit+x] = new_elem_ptr_[n*Nunit+x];
      }
    }

    template <class T1, class T2>
    void SetElems_conti_cpu_impl(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit) {
      // Start copy elem:
      T1 *new_elem_ptr_ = static_cast<T1 *>(in);
      T2 *elem_ptr_ = static_cast<T2 *>(out);

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
        for (cytnx_uint64 x = 0; x < Nunit; x++)
          elem_ptr_[Loc * Nunit + x] = new_elem_ptr_[n * Nunit + x];
      }
    }

    template <class T1, class T2>
    void SetElems_conti_cpu_scal_impl(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                      const std::vector<cytnx_uint64> &new_offj,
                                      const std::vector<std::vector<cytnx_uint64>> &locators,
                                      const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit) {
      // Start copy elem:
      T1 new_elem_ = *(static_cast<T1 *>(in));
      T2 *elem_ptr_ = static_cast<T2 *>(out);

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
        for (cytnx_uint64 x = 0; x < Nunit; x++) elem_ptr_[Loc * Nunit + x] = new_elem_;
      }
    }

    // out is the target Tensor, in is the rhs
    void SetElems_conti_cpu_cdtcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_complex128, cytnx_complex128>(
          in, out, offj, new_offj, locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl_sametype<cytnx_complex128>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_cdtcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_complex128, cytnx_complex64>(in, out, offj, new_offj,
                                                                        locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_complex128, cytnx_complex64>(in, out, offj, new_offj,
                                                                   locators, TotalElem, Nunit);
    }

    //----
    void SetElems_conti_cpu_cftcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_complex64, cytnx_complex128>(in, out, offj, new_offj,
                                                                        locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_complex64, cytnx_complex128>(in, out, offj, new_offj,
                                                                   locators, TotalElem, Nunit);
    }
    void SetElems_conti_cpu_cftcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_complex64, cytnx_complex64>(in, out, offj, new_offj,
                                                                       locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl_sametype<cytnx_complex64>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }

    //----
    void SetElems_conti_cpu_dtcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_double, cytnx_complex128>(in, out, offj, new_offj,
                                                                     locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_double, cytnx_complex128>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
    }
    void SetElems_conti_cpu_dtcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_double, cytnx_complex64>(in, out, offj, new_offj,
                                                                    locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_double, cytnx_complex64>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
    }
    void SetElems_conti_cpu_dtd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_double, cytnx_double>(in, out, offj, new_offj, locators,
                                                                 TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl_sametype<cytnx_double>(in, out, offj, new_offj, locators, TotalElem,
                                                       Nunit);
    }
    void SetElems_conti_cpu_dtf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_double, cytnx_float>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_double, cytnx_float>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_dti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_double, cytnx_int64>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_double, cytnx_int64>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_dtu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_double, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                                 TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_double, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                            TotalElem, Nunit);
    }
    void SetElems_conti_cpu_dti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_double, cytnx_int32>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_double, cytnx_int32>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_dtu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_double, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                                 TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_double, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                            TotalElem, Nunit);
    }
    void SetElems_conti_cpu_dti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_double, cytnx_int16>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_double, cytnx_int16>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_dtu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_double, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                                 TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_double, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                            TotalElem, Nunit);
    }
    void SetElems_conti_cpu_dtb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_double, cytnx_bool>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_double, cytnx_bool>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }

    //----
    void SetElems_conti_cpu_ftcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_float, cytnx_complex128>(in, out, offj, new_offj,
                                                                    locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_float, cytnx_complex128>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
    }
    void SetElems_conti_cpu_ftcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_float, cytnx_complex64>(in, out, offj, new_offj,
                                                                   locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_float, cytnx_complex64>(in, out, offj, new_offj, locators,
                                                              TotalElem, Nunit);
    }
    void SetElems_conti_cpu_ftd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_float, cytnx_double>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_float, cytnx_double>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_ftf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_float, cytnx_float>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl_sametype<cytnx_float>(in, out, offj, new_offj, locators, TotalElem,
                                                      Nunit);
    }
    void SetElems_conti_cpu_fti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_float, cytnx_int64>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_float, cytnx_int64>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }
    void SetElems_conti_cpu_ftu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_float, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_float, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_fti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_float, cytnx_int32>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_float, cytnx_int32>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }
    void SetElems_conti_cpu_ftu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_float, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_float, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_fti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_float, cytnx_int16>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_float, cytnx_int16>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }
    void SetElems_conti_cpu_ftu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_float, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_float, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_ftb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_float, cytnx_bool>(in, out, offj, new_offj, locators,
                                                              TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_float, cytnx_bool>(in, out, offj, new_offj, locators,
                                                         TotalElem, Nunit);
    }

    //----
    void SetElems_conti_cpu_i64tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int64, cytnx_complex128>(in, out, offj, new_offj,
                                                                    locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int64, cytnx_complex128>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i64tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int64, cytnx_complex64>(in, out, offj, new_offj,
                                                                   locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int64, cytnx_complex64>(in, out, offj, new_offj, locators,
                                                              TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i64td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int64, cytnx_double>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int64, cytnx_double>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i64tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int64, cytnx_float>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int64, cytnx_float>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i64ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int64, cytnx_int64>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl_sametype<cytnx_int64>(in, out, offj, new_offj, locators, TotalElem,
                                                      Nunit);
    }
    void SetElems_conti_cpu_i64tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int64, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int64, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i64ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int64, cytnx_int32>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int64, cytnx_int32>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i64tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int64, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int64, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i64ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int64, cytnx_int16>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int64, cytnx_int16>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i64tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int64, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int64, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i64tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int64, cytnx_bool>(in, out, offj, new_offj, locators,
                                                              TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int64, cytnx_bool>(in, out, offj, new_offj, locators,
                                                         TotalElem, Nunit);
    }

    //----
    void SetElems_conti_cpu_u64tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint64, cytnx_complex128>(in, out, offj, new_offj,
                                                                     locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint64, cytnx_complex128>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u64tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint64, cytnx_complex64>(in, out, offj, new_offj,
                                                                    locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint64, cytnx_complex64>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u64td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint64, cytnx_double>(in, out, offj, new_offj, locators,
                                                                 TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint64, cytnx_double>(in, out, offj, new_offj, locators,
                                                            TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u64tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint64, cytnx_float>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint64, cytnx_float>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u64ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint64, cytnx_int64>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint64, cytnx_int64>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u64tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint64, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                                 TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl_sametype<cytnx_uint64>(in, out, offj, new_offj, locators, TotalElem,
                                                       Nunit);
    }
    void SetElems_conti_cpu_u64ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint64, cytnx_int32>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint64, cytnx_int32>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u64tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint64, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                                 TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint64, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                            TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u64ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint64, cytnx_int16>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint64, cytnx_int16>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u64tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint64, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                                 TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint64, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                            TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u64tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint64, cytnx_bool>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint64, cytnx_bool>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }

    //----
    void SetElems_conti_cpu_i32tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int32, cytnx_complex128>(in, out, offj, new_offj,
                                                                    locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int32, cytnx_complex128>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i32tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int32, cytnx_complex64>(in, out, offj, new_offj,
                                                                   locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int32, cytnx_complex64>(in, out, offj, new_offj, locators,
                                                              TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i32td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int32, cytnx_double>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int32, cytnx_double>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i32tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int32, cytnx_float>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int32, cytnx_float>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i32ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int32, cytnx_int64>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int32, cytnx_int64>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i32tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int32, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int32, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i32ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int32, cytnx_int32>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl_sametype<cytnx_int32>(in, out, offj, new_offj, locators, TotalElem,
                                                      Nunit);
    }
    void SetElems_conti_cpu_i32tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int32, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int32, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i32ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int32, cytnx_int16>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int32, cytnx_int16>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i32tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int32, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int32, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i32tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int32, cytnx_bool>(in, out, offj, new_offj, locators,
                                                              TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int32, cytnx_bool>(in, out, offj, new_offj, locators,
                                                         TotalElem, Nunit);
    }

    //----
    void SetElems_conti_cpu_u32tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint32, cytnx_complex128>(in, out, offj, new_offj,
                                                                     locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint32, cytnx_complex128>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u32tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint32, cytnx_complex64>(in, out, offj, new_offj,
                                                                    locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint32, cytnx_complex64>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u32td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint32, cytnx_double>(in, out, offj, new_offj, locators,
                                                                 TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint32, cytnx_double>(in, out, offj, new_offj, locators,
                                                            TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u32tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint32, cytnx_float>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint32, cytnx_float>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u32ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint32, cytnx_int64>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint32, cytnx_int64>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u32tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint32, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                                 TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint32, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                            TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u32ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint32, cytnx_int32>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint32, cytnx_int32>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u32tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint32, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                                 TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl_sametype<cytnx_uint32>(in, out, offj, new_offj, locators, TotalElem,
                                                       Nunit);
    }
    void SetElems_conti_cpu_u32ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint32, cytnx_int16>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint32, cytnx_int16>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u32tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint32, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                                 TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint32, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                            TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u32tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint32, cytnx_bool>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint32, cytnx_bool>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }

    //----
    void SetElems_conti_cpu_u16tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint16, cytnx_complex128>(in, out, offj, new_offj,
                                                                     locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint16, cytnx_complex128>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u16tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint16, cytnx_complex64>(in, out, offj, new_offj,
                                                                    locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint16, cytnx_complex64>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u16td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint16, cytnx_double>(in, out, offj, new_offj, locators,
                                                                 TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint16, cytnx_double>(in, out, offj, new_offj, locators,
                                                            TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u16tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint16, cytnx_float>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint16, cytnx_float>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u16ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint16, cytnx_int64>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint16, cytnx_int64>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u16tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint16, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                                 TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint16, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                            TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u16ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint16, cytnx_int32>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint16, cytnx_int32>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u16tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint16, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                                 TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint16, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                            TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u16ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint16, cytnx_int16>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint16, cytnx_int16>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_u16tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint16, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                                 TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl_sametype<cytnx_uint16>(in, out, offj, new_offj, locators, TotalElem,
                                                       Nunit);
    }
    void SetElems_conti_cpu_u16tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_uint16, cytnx_bool>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_uint16, cytnx_bool>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }

    //----
    void SetElems_conti_cpu_i16tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int16, cytnx_complex128>(in, out, offj, new_offj,
                                                                    locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int16, cytnx_complex128>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i16tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int16, cytnx_complex64>(in, out, offj, new_offj,
                                                                   locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int16, cytnx_complex64>(in, out, offj, new_offj, locators,
                                                              TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i16td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int16, cytnx_double>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int16, cytnx_double>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i16tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int16, cytnx_float>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int16, cytnx_float>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i16ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int16, cytnx_int64>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int16, cytnx_int64>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i16tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int16, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int16, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i16ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int16, cytnx_int32>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int16, cytnx_int32>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i16tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int16, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int16, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i16ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int16, cytnx_int16>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl_sametype<cytnx_int16>(in, out, offj, new_offj, locators, TotalElem,
                                                      Nunit);
    }
    void SetElems_conti_cpu_i16tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int16, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                                TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int16, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                           TotalElem, Nunit);
    }
    void SetElems_conti_cpu_i16tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_int16, cytnx_bool>(in, out, offj, new_offj, locators,
                                                              TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_int16, cytnx_bool>(in, out, offj, new_offj, locators,
                                                         TotalElem, Nunit);
    }

    //----
    void SetElems_conti_cpu_btcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_bool, cytnx_complex128>(in, out, offj, new_offj,
                                                                   locators, TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_bool, cytnx_complex128>(in, out, offj, new_offj, locators,
                                                              TotalElem, Nunit);
    }
    void SetElems_conti_cpu_btcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_bool, cytnx_complex64>(in, out, offj, new_offj, locators,
                                                                  TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_bool, cytnx_complex64>(in, out, offj, new_offj, locators,
                                                             TotalElem, Nunit);
    }
    void SetElems_conti_cpu_btd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_bool, cytnx_double>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_bool, cytnx_double>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }
    void SetElems_conti_cpu_btf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_bool, cytnx_float>(in, out, offj, new_offj, locators,
                                                              TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_bool, cytnx_float>(in, out, offj, new_offj, locators,
                                                         TotalElem, Nunit);
    }
    void SetElems_conti_cpu_bti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_bool, cytnx_int64>(in, out, offj, new_offj, locators,
                                                              TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_bool, cytnx_int64>(in, out, offj, new_offj, locators,
                                                         TotalElem, Nunit);
    }
    void SetElems_conti_cpu_btu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_bool, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_bool, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }
    void SetElems_conti_cpu_bti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_bool, cytnx_int32>(in, out, offj, new_offj, locators,
                                                              TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_bool, cytnx_int32>(in, out, offj, new_offj, locators,
                                                         TotalElem, Nunit);
    }
    void SetElems_conti_cpu_btu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_bool, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_bool, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }
    void SetElems_conti_cpu_bti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_bool, cytnx_int16>(in, out, offj, new_offj, locators,
                                                              TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_bool, cytnx_int16>(in, out, offj, new_offj, locators,
                                                         TotalElem, Nunit);
    }
    void SetElems_conti_cpu_btu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_bool, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                               TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl<cytnx_bool, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                          TotalElem, Nunit);
    }
    void SetElems_conti_cpu_btb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar) {
      if (is_scalar)
        SetElems_conti_cpu_scal_impl<cytnx_bool, cytnx_bool>(in, out, offj, new_offj, locators,
                                                             TotalElem, Nunit);
      else
        SetElems_conti_cpu_impl_sametype<cytnx_bool>(in, out, offj, new_offj, locators, TotalElem,
                                                     Nunit);
    }

  }  // namespace utils_internal
}  // namespace cytnx
