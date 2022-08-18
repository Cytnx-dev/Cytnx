#include "utils/utils_internal_cpu/Movemem_cpu.hpp"
#include "Storage.hpp"
#ifdef UNI_OMP
  #include <omp.h>
#endif

#ifdef UNI_HPTT
  #include "hptt.h"
#endif

using namespace std;
namespace cytnx {

  namespace utils_internal {

    boost::intrusive_ptr<Storage_base> Movemem_cpu_cd(boost::intrusive_ptr<Storage_base> &in,
                                                      const std::vector<cytnx_uint64> &old_shape,
                                                      const std::vector<cytnx_uint64> &mapper,
                                                      const std::vector<cytnx_uint64> &invmapper,
                                                      const bool is_inplace) {
#ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.ComplexDouble,
        "[DEBUG][internal error] in.dtype_str is [%s] but call Movemem_cpu with type ComplexDouble",
        in->dtype_str().c_str());
#endif

      cytnx_complex128 *des = (cytnx_complex128 *)malloc(in->cap * sizeof(cytnx_complex128));
      cytnx_complex128 *src = static_cast<cytnx_complex128 *>(in->Mem);
      cytnx_uint64 accu_old = 1, accu_new = 1;

#ifdef UNI_HPTT
  #ifdef UNI_DEBUG
      cytnx_error_msg(true, "[DEBUG][Internal prompt] USE HPTT%s", "\n");
  #endif
      std::vector<int> perm(mapper.begin(), mapper.end());
      std::vector<int> size(old_shape.begin(), old_shape.end());
      auto plan = hptt::create_plan(&perm[0], perm.size(), 1, src, &size[0], NULL, 0, des, NULL,
                                    hptt::ESTIMATE, cytnx::Device.Ncpus, nullptr, true);
      plan->execute();
      accu_old = in->size();

#else

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }

  #ifdef UNI_OMP
      std::vector<std::vector<cytnx_uint64>> old_inds;
    #pragma omp parallel
      {
        if (omp_get_thread_num() == 0)
          old_inds = std::vector<std::vector<cytnx_uint64>>(
            omp_get_num_threads(), std::vector<cytnx_uint64>(old_shape.size()));
      }

    #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        cytnx_uint64 j;
        cytnx_uint64 old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[omp_get_thread_num()][j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[omp_get_thread_num()][mapper[j]];
        }
        des[old_loc] = src[n];
      }

  #else
      std::vector<cytnx_uint64> old_inds(old_shape.size());
      cytnx_uint64 j, old_loc;
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[mapper[j]];
        }
        des[old_loc] = src[n];
      }
  #endif
#endif  // hptt

      boost::intrusive_ptr<Storage_base> out(new ComplexDoubleStorage());
      if (is_inplace) {
        memcpy(in->Mem, des, sizeof(cytnx_complex128) * accu_old);
        free(des);
        return out;
      } else {
        out->_Init_byptr(des, accu_old, in->device, true, in->cap);
        return out;
      }
    }

    boost::intrusive_ptr<Storage_base> Movemem_cpu_cf(boost::intrusive_ptr<Storage_base> &in,
                                                      const std::vector<cytnx_uint64> &old_shape,
                                                      const std::vector<cytnx_uint64> &mapper,
                                                      const std::vector<cytnx_uint64> &invmapper,
                                                      const bool is_inplace) {
#ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.ComplexFloat,
        "[DEBUG][internal error] in.dtype_str is [%s] but call Movemem_cpu with type ComplexFloat",
        in->dtype_str().c_str());
#endif

      cytnx_complex64 *des = (cytnx_complex64 *)malloc(in->cap * sizeof(cytnx_complex64));
      cytnx_complex64 *src = static_cast<cytnx_complex64 *>(in->Mem);
      cytnx_uint64 accu_old = 1, accu_new = 1;

#ifdef UNI_HPTT
  #ifdef UNI_DEBUG
      cytnx_error_msg(true, "[DEBUG][Internal prompt] USE HPTT%s", "\n");
  #endif
      std::vector<int> perm(mapper.begin(), mapper.end());
      std::vector<int> size(old_shape.begin(), old_shape.end());
      auto plan = hptt::create_plan(&perm[0], perm.size(), 1, src, &size[0], NULL, 0, des, NULL,
                                    hptt::ESTIMATE, cytnx::Device.Ncpus, nullptr, true);
      plan->execute();
      accu_old = in->size();

#else

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }

  #ifdef UNI_OMP
      std::vector<std::vector<cytnx_uint64>> old_inds;
    #pragma omp parallel
      {
        if (omp_get_thread_num() == 0)
          old_inds = std::vector<std::vector<cytnx_uint64>>(
            omp_get_num_threads(), std::vector<cytnx_uint64>(old_shape.size()));
      }

    #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        cytnx_uint64 j;
        cytnx_uint64 old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[omp_get_thread_num()][j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[omp_get_thread_num()][mapper[j]];
        }
        des[old_loc] = src[n];
      }

  #else
      std::vector<cytnx_uint64> old_inds(old_shape.size());
      cytnx_uint64 j, old_loc;
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[mapper[j]];
        }
        des[old_loc] = src[n];
      }
  #endif
#endif  // hptt

      boost::intrusive_ptr<Storage_base> out(new ComplexFloatStorage());
      if (is_inplace) {
        memcpy(in->Mem, des, sizeof(cytnx_complex64) * accu_old);
        free(des);
        return out;
      } else {
        out->_Init_byptr(des, accu_old, in->device, true, in->cap);
        return out;
      }
    }

    boost::intrusive_ptr<Storage_base> Movemem_cpu_d(boost::intrusive_ptr<Storage_base> &in,
                                                     const std::vector<cytnx_uint64> &old_shape,
                                                     const std::vector<cytnx_uint64> &mapper,
                                                     const std::vector<cytnx_uint64> &invmapper,
                                                     const bool is_inplace) {
#ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Double,
        "[DEBUG][internal error] in.dtype_str is [%s] but call Movemem_cpu with type Double",
        in->dtype_str().c_str());
#endif

      cytnx_double *des = (cytnx_double *)calloc(in->cap, sizeof(cytnx_double));
      cytnx_double *src = static_cast<cytnx_double *>(in->Mem);
      cytnx_uint64 accu_old = 1, accu_new = 1;

#ifdef UNI_HPTT
  #ifdef UNI_DEBUG
      cytnx_error_msg(true, "[DEBUG][Internal prompt] USE HPTT%s", "\n");
  #endif
      std::vector<int> perm(mapper.begin(), mapper.end());
      std::vector<int> size(old_shape.begin(), old_shape.end());
      auto plan = hptt::create_plan(&perm[0], perm.size(), 1, src, &size[0], NULL, 0, des, NULL,
                                    hptt::ESTIMATE, cytnx::Device.Ncpus, nullptr, true);
      plan->execute();
      accu_old = in->size();

#else

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }

  #ifdef UNI_OMP
      std::vector<std::vector<cytnx_uint64>> old_inds;
    #pragma omp parallel
      {
        if (omp_get_thread_num() == 0)
          old_inds = std::vector<std::vector<cytnx_uint64>>(
            omp_get_num_threads(), std::vector<cytnx_uint64>(old_shape.size()));
      }

    #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        cytnx_uint64 j;
        cytnx_uint64 old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[omp_get_thread_num()][j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[omp_get_thread_num()][mapper[j]];
        }
        des[old_loc] = src[n];
      }

  #else
      std::vector<cytnx_uint64> old_inds(old_shape.size());
      cytnx_uint64 j, old_loc;
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[mapper[j]];
        }
        des[old_loc] = src[n];
      }
  #endif

#endif  // HPTT

      boost::intrusive_ptr<Storage_base> out(new DoubleStorage());
      if (is_inplace) {
        memcpy(in->Mem, des, sizeof(cytnx_double) * accu_old);
        free(des);
        return out;
      } else {
        out->_Init_byptr(des, accu_old, in->device, true, in->cap);
        return out;
      }
    }

    boost::intrusive_ptr<Storage_base> Movemem_cpu_f(boost::intrusive_ptr<Storage_base> &in,
                                                     const std::vector<cytnx_uint64> &old_shape,
                                                     const std::vector<cytnx_uint64> &mapper,
                                                     const std::vector<cytnx_uint64> &invmapper,
                                                     const bool is_inplace) {
#ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Float,
        "[DEBUG][internal error] in.dtype_str is [%s] but call Movemem_cpu with type Float",
        in->dtype_str().c_str());
#endif

      cytnx_float *des = (cytnx_float *)malloc(in->cap * sizeof(cytnx_float));
      cytnx_float *src = static_cast<cytnx_float *>(in->Mem);
      cytnx_uint64 accu_old = 1, accu_new = 1;
#ifdef UNI_HPTT
  #ifdef UNI_DEBUG
      cytnx_error_msg(true, "[DEBUG][Internal prompt] USE HPTT%s", "\n");
  #endif
      std::vector<int> perm(mapper.begin(), mapper.end());
      std::vector<int> size(old_shape.begin(), old_shape.end());
      auto plan = hptt::create_plan(&perm[0], perm.size(), 1, src, &size[0], NULL, 0, des, NULL,
                                    hptt::ESTIMATE, cytnx::Device.Ncpus, nullptr, true);
      plan->execute();
      accu_old = in->size();
#else

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }

  #ifdef UNI_OMP
      std::vector<std::vector<cytnx_uint64>> old_inds;
    #pragma omp parallel
      {
        if (omp_get_thread_num() == 0)
          old_inds = std::vector<std::vector<cytnx_uint64>>(
            omp_get_num_threads(), std::vector<cytnx_uint64>(old_shape.size()));
      }

    #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        cytnx_uint64 j;
        cytnx_uint64 old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[omp_get_thread_num()][j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[omp_get_thread_num()][mapper[j]];
        }
        des[old_loc] = src[n];
      }

  #else
      std::vector<cytnx_uint64> old_inds(old_shape.size());
      cytnx_uint64 j, old_loc;
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[mapper[j]];
        }
        des[old_loc] = src[n];
      }
  #endif
#endif  // hptt

      boost::intrusive_ptr<Storage_base> out(new FloatStorage());
      if (is_inplace) {
        memcpy(in->Mem, des, sizeof(cytnx_float) * accu_old);
        free(des);
        return out;
      } else {
        out->_Init_byptr(des, accu_old, in->device, true, in->cap);
        return out;
      }
    }

    boost::intrusive_ptr<Storage_base> Movemem_cpu_i64(boost::intrusive_ptr<Storage_base> &in,
                                                       const std::vector<cytnx_uint64> &old_shape,
                                                       const std::vector<cytnx_uint64> &mapper,
                                                       const std::vector<cytnx_uint64> &invmapper,
                                                       const bool is_inplace) {
#ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Int64,
        "[DEBUG][internal error] in.dtype_str is [%s] but call Movemem_cpu with type Int64",
        in->dtype_str().c_str());
#endif

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }

      cytnx_int64 *des = (cytnx_int64 *)malloc(in->cap * sizeof(cytnx_int64));
      cytnx_int64 *src = static_cast<cytnx_int64 *>(in->Mem);

#ifdef UNI_OMP
      std::vector<std::vector<cytnx_uint64>> old_inds;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0)
          old_inds = std::vector<std::vector<cytnx_uint64>>(
            omp_get_num_threads(), std::vector<cytnx_uint64>(old_shape.size()));
      }

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        cytnx_uint64 j;
        cytnx_uint64 old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[omp_get_thread_num()][j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[omp_get_thread_num()][mapper[j]];
        }
        des[old_loc] = src[n];
      }

#else
      std::vector<cytnx_uint64> old_inds(old_shape.size());
      cytnx_uint64 j, old_loc;
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[mapper[j]];
        }
        des[old_loc] = src[n];
      }
#endif

      boost::intrusive_ptr<Storage_base> out(new Int64Storage());
      if (is_inplace) {
        memcpy(in->Mem, des, sizeof(cytnx_int64) * accu_old);
        free(des);
        return out;
      } else {
        out->_Init_byptr(des, accu_old, in->device, true, in->cap);
        return out;
      }
    }

    boost::intrusive_ptr<Storage_base> Movemem_cpu_u64(boost::intrusive_ptr<Storage_base> &in,
                                                       const std::vector<cytnx_uint64> &old_shape,
                                                       const std::vector<cytnx_uint64> &mapper,
                                                       const std::vector<cytnx_uint64> &invmapper,
                                                       const bool is_inplace) {
#ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Uint64,
        "[DEBUG][internal error] in.dtype_str is [%s] but call Movemem_cpu with type Uint64",
        in->dtype_str().c_str());
#endif

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }

      cytnx_uint64 *des = (cytnx_uint64 *)malloc(in->cap * sizeof(cytnx_uint64));
      cytnx_uint64 *src = static_cast<cytnx_uint64 *>(in->Mem);

#ifdef UNI_OMP
      std::vector<std::vector<cytnx_uint64>> old_inds;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0)
          old_inds = std::vector<std::vector<cytnx_uint64>>(
            omp_get_num_threads(), std::vector<cytnx_uint64>(old_shape.size()));
      }

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        cytnx_uint64 j;
        cytnx_uint64 old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[omp_get_thread_num()][j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[omp_get_thread_num()][mapper[j]];
        }
        des[old_loc] = src[n];
      }

#else
      std::vector<cytnx_uint64> old_inds(old_shape.size());
      cytnx_uint64 j, old_loc;
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[mapper[j]];
        }
        des[old_loc] = src[n];
      }
#endif

      boost::intrusive_ptr<Storage_base> out(new Uint64Storage());
      if (is_inplace) {
        memcpy(in->Mem, des, sizeof(cytnx_uint64) * accu_old);
        free(des);
        return out;
      } else {
        out->_Init_byptr(des, accu_old, in->device, true, in->cap);
        return out;
      }
    }

    boost::intrusive_ptr<Storage_base> Movemem_cpu_i32(boost::intrusive_ptr<Storage_base> &in,
                                                       const std::vector<cytnx_uint64> &old_shape,
                                                       const std::vector<cytnx_uint64> &mapper,
                                                       const std::vector<cytnx_uint64> &invmapper,
                                                       const bool is_inplace) {
#ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Int32,
        "[DEBUG][internal error] in.dtype_str is [%s] but call Movemem_cpu with type Int32",
        in->dtype_str().c_str());
#endif

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }

      cytnx_int32 *des = (cytnx_int32 *)malloc(in->cap * sizeof(cytnx_int32));
      cytnx_int32 *src = static_cast<cytnx_int32 *>(in->Mem);

#ifdef UNI_OMP
      std::vector<std::vector<cytnx_uint64>> old_inds;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0)
          old_inds = std::vector<std::vector<cytnx_uint64>>(
            omp_get_num_threads(), std::vector<cytnx_uint64>(old_shape.size()));
      }

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        cytnx_uint64 j;
        cytnx_uint64 old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[omp_get_thread_num()][j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[omp_get_thread_num()][mapper[j]];
        }
        des[old_loc] = src[n];
      }

#else
      std::vector<cytnx_uint64> old_inds(old_shape.size());
      cytnx_uint64 j, old_loc;
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[mapper[j]];
        }
        des[old_loc] = src[n];
      }
#endif

      boost::intrusive_ptr<Storage_base> out(new Int32Storage());
      if (is_inplace) {
        memcpy(in->Mem, des, sizeof(cytnx_int32) * accu_old);
        free(des);
        return out;
      } else {
        out->_Init_byptr(des, accu_old, in->device, true, in->cap);
        return out;
      }
    }

    boost::intrusive_ptr<Storage_base> Movemem_cpu_u32(boost::intrusive_ptr<Storage_base> &in,
                                                       const std::vector<cytnx_uint64> &old_shape,
                                                       const std::vector<cytnx_uint64> &mapper,
                                                       const std::vector<cytnx_uint64> &invmapper,
                                                       const bool is_inplace) {
#ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Uint32,
        "[DEBUG][internal error] in.dtype_str is [%s] but call Movemem_cpu with type Uint32",
        in->dtype_str().c_str());
#endif

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }

      cytnx_uint32 *des = (cytnx_uint32 *)malloc(in->cap * sizeof(cytnx_uint32));
      cytnx_uint32 *src = static_cast<cytnx_uint32 *>(in->Mem);

#ifdef UNI_OMP
      std::vector<std::vector<cytnx_uint64>> old_inds;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0)
          old_inds = std::vector<std::vector<cytnx_uint64>>(
            omp_get_num_threads(), std::vector<cytnx_uint64>(old_shape.size()));
      }

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        cytnx_uint64 j;
        cytnx_uint64 old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[omp_get_thread_num()][j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[omp_get_thread_num()][mapper[j]];
        }
        des[old_loc] = src[n];
      }

#else
      std::vector<cytnx_uint64> old_inds(old_shape.size());
      cytnx_uint64 j, old_loc;
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[mapper[j]];
        }
        des[old_loc] = src[n];
      }
#endif

      boost::intrusive_ptr<Storage_base> out(new Uint32Storage());
      if (is_inplace) {
        memcpy(in->Mem, des, sizeof(cytnx_uint32) * accu_old);
        free(des);
        return out;
      } else {
        out->_Init_byptr(des, accu_old, in->device, true, in->cap);
        return out;
      }
    }
    boost::intrusive_ptr<Storage_base> Movemem_cpu_u16(boost::intrusive_ptr<Storage_base> &in,
                                                       const std::vector<cytnx_uint64> &old_shape,
                                                       const std::vector<cytnx_uint64> &mapper,
                                                       const std::vector<cytnx_uint64> &invmapper,
                                                       const bool is_inplace) {
#ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Uint16,
        "[DEBUG][internal error] in.dtype_str is [%s] but call Movemem_cpu with type Uint16",
        in->dtype_str().c_str());
#endif

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }

      cytnx_uint16 *des = (cytnx_uint16 *)malloc(in->cap * sizeof(cytnx_uint16));
      cytnx_uint16 *src = static_cast<cytnx_uint16 *>(in->Mem);

#ifdef UNI_OMP
      std::vector<std::vector<cytnx_uint64>> old_inds;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0)
          old_inds = std::vector<std::vector<cytnx_uint64>>(
            omp_get_num_threads(), std::vector<cytnx_uint64>(old_shape.size()));
      }

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        cytnx_uint64 j;
        cytnx_uint64 old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[omp_get_thread_num()][j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[omp_get_thread_num()][mapper[j]];
        }
        des[old_loc] = src[n];
      }

#else
      std::vector<cytnx_uint64> old_inds(old_shape.size());
      cytnx_uint64 j, old_loc;
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[mapper[j]];
        }
        des[old_loc] = src[n];
      }
#endif

      boost::intrusive_ptr<Storage_base> out(new Uint16Storage());
      if (is_inplace) {
        memcpy(in->Mem, des, sizeof(cytnx_uint16) * accu_old);
        free(des);
        return out;
      } else {
        out->_Init_byptr(des, accu_old, in->device, true, in->cap);
        return out;
      }
    }
    boost::intrusive_ptr<Storage_base> Movemem_cpu_i16(boost::intrusive_ptr<Storage_base> &in,
                                                       const std::vector<cytnx_uint64> &old_shape,
                                                       const std::vector<cytnx_uint64> &mapper,
                                                       const std::vector<cytnx_uint64> &invmapper,
                                                       const bool is_inplace) {
#ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Int16,
        "[DEBUG][internal error] in.dtype_str is [%s] but call Movemem_cpu with type Int16",
        in->dtype_str().c_str());
#endif

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }

      cytnx_int16 *des = (cytnx_int16 *)malloc(in->cap * sizeof(cytnx_int16));
      cytnx_int16 *src = static_cast<cytnx_int16 *>(in->Mem);

#ifdef UNI_OMP
      std::vector<std::vector<cytnx_uint64>> old_inds;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0)
          old_inds = std::vector<std::vector<cytnx_uint64>>(
            omp_get_num_threads(), std::vector<cytnx_uint64>(old_shape.size()));
      }

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        cytnx_uint64 j;
        cytnx_uint64 old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[omp_get_thread_num()][j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[omp_get_thread_num()][mapper[j]];
        }
        des[old_loc] = src[n];
      }

#else
      std::vector<cytnx_uint64> old_inds(old_shape.size());
      cytnx_uint64 j, old_loc;
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[mapper[j]];
        }
        des[old_loc] = src[n];
      }
#endif

      boost::intrusive_ptr<Storage_base> out(new Int16Storage());
      if (is_inplace) {
        memcpy(in->Mem, des, sizeof(cytnx_int16) * accu_old);
        free(des);
        return out;
      } else {
        out->_Init_byptr(des, accu_old, in->device, true, in->cap);
        return out;
      }
    }
    boost::intrusive_ptr<Storage_base> Movemem_cpu_b(boost::intrusive_ptr<Storage_base> &in,
                                                     const std::vector<cytnx_uint64> &old_shape,
                                                     const std::vector<cytnx_uint64> &mapper,
                                                     const std::vector<cytnx_uint64> &invmapper,
                                                     const bool is_inplace) {
#ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Bool,
        "[DEBUG][internal error] in.dtype_str is [%s] but call Movemem_cpu with type Bool",
        in->dtype_str().c_str());
#endif

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }

      cytnx_bool *des = (cytnx_bool *)malloc(in->cap * sizeof(cytnx_bool));
      cytnx_bool *src = static_cast<cytnx_bool *>(in->Mem);

#ifdef UNI_OMP
      std::vector<std::vector<cytnx_uint64>> old_inds;
  #pragma omp parallel
      {
        if (omp_get_thread_num() == 0)
          old_inds = std::vector<std::vector<cytnx_uint64>>(
            omp_get_num_threads(), std::vector<cytnx_uint64>(old_shape.size()));
      }

  #pragma omp parallel for schedule(dynamic)
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        cytnx_uint64 j;
        cytnx_uint64 old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[omp_get_thread_num()][j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[omp_get_thread_num()][mapper[j]];
        }
        des[old_loc] = src[n];
      }

#else
      std::vector<cytnx_uint64> old_inds(old_shape.size());
      cytnx_uint64 j, old_loc;
      for (cytnx_uint64 n = 0; n < accu_old; n++) {
        // calc new id:
        old_loc = n;
        for (j = 0; j < old_shape.size(); j++) {
          old_inds[j] = old_loc / shifter_old[j];
          old_loc = old_loc % shifter_old[j];
        }
        old_loc = 0;  // position:
        for (j = 0; j < old_shape.size(); j++) {
          old_loc += shifter_new[j] * old_inds[mapper[j]];
        }
        des[old_loc] = src[n];
      }
#endif

      boost::intrusive_ptr<Storage_base> out(new BoolStorage());
      if (is_inplace) {
        memcpy(in->Mem, des, sizeof(cytnx_bool) * accu_old);
        free(des);
        return out;
      } else {
        out->_Init_byptr(des, accu_old, in->device, true, in->cap);
        return out;
      }
    }

  }  // namespace utils_internal
}  // namespace cytnx
