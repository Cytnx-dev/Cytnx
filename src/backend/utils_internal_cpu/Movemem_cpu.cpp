#include "Movemem_cpu.hpp"

#include <type_traits>
#include <vector>

#include "boost/smart_ptr/intrusive_ptr.hpp"

#include "backend/Storage.hpp"

#ifdef UNI_OMP
  #include <omp.h>
#endif

#ifdef UNI_HPTT
  #include "hptt.h"
#endif

using namespace std;
namespace cytnx {

  namespace utils_internal {

    template <typename T, typename std::enable_if_t<std::is_integral_v<T>, bool>>
    boost::intrusive_ptr<Storage_base> MoveMemoryCpu(boost::intrusive_ptr<Storage_base> &in,
                                                     const std::vector<cytnx_uint64> &old_shape,
                                                     const std::vector<cytnx_uint64> &mapper,
                                                     const std::vector<cytnx_uint64> &invmapper,
                                                     const bool is_inplace) {
#ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.cy_typeid(T()),
        "[DEBUG][internal error] in.dtype_str is [%s] but call MoveMemoryCpu with type %s",
        in->dtype_str().c_str(), Type.getname(Type.cy_typeid(T())));
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

      T *des = (T *)malloc(in->cap * sizeof(T));
      T *src = static_cast<T *>(in->Mem);

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
      boost::intrusive_ptr<Storage_base> out;
      if constexpr (std::is_same_v<T, cytnx_uint64>) {
        out = new Uint64Storage();
      } else if (std::is_same_v<T, cytnx_int64>) {
        out = new Int64Storage();
      } else if (std::is_same_v<T, cytnx_uint32>) {
        out = new Uint32Storage();
      } else if (std::is_same_v<T, cytnx_int32>) {
        out = new Int32Storage();
      } else if (std::is_same_v<T, cytnx_uint16>) {
        out = new Uint16Storage();
      } else if (std::is_same_v<T, cytnx_int16>) {
        out = new Int16Storage();
      } else if (std::is_same_v<T, cytnx_bool>) {
        out = new BoolStorage();
      }
      if (is_inplace) {
        memcpy(in->Mem, des, sizeof(T) * accu_old);
        free(des);
        return out;
      } else {
        out->_Init_byptr(des, accu_old, in->device, true, in->cap);
        return out;
      }
    }

    template <typename T, typename std::enable_if_t<!std::is_integral_v<T>, bool>>
    boost::intrusive_ptr<Storage_base> MoveMemoryCpu(boost::intrusive_ptr<Storage_base> &in,
                                                     const std::vector<cytnx_uint64> &old_shape,
                                                     const std::vector<cytnx_uint64> &mapper,
                                                     const std::vector<cytnx_uint64> &invmapper,
                                                     const bool is_inplace) {
#ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.cy_typeid(T()),
        "[DEBUG][internal error] in.dtype_str is [%s] but call MoveMemoryCpu with type %s",
        in->dtype_str().c_str(), Type.getname(Type.cy_typeid(T())));
#endif

      T *des = (T *)malloc(in->cap * sizeof(T));
      T *src = static_cast<T *>(in->Mem);
      cytnx_uint64 accu_old = 1, accu_new = 1;

#ifdef UNI_HPTT
  #ifdef UNI_DEBUG
      cytnx_error_msg(true, "[DEBUG][Internal prompt] USE HPTT%s", "\n");
  #endif
      if (in->size() > 64) {
        std::vector<int> perm(mapper.begin(), mapper.end());
        std::vector<int> size(old_shape.begin(), old_shape.end());
        auto plan = hptt::create_plan(&perm[0], perm.size(), 1, src, &size[0], NULL, 0, des, NULL,
                                      hptt::ESTIMATE, cytnx::Device.Ncpus, nullptr, true);
        plan->execute();
        accu_old = in->size();
      } else {
        std::vector<cytnx_int64> newshape(old_shape.size());
        for (cytnx_int64 i = 0; i < old_shape.size(); i++) {
          newshape[i] = old_shape[mapper[i]];
        }
        std::vector<cytnx_int64> shifter_new(old_shape.size());

        for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
          shifter_new[i] = accu_new;
          accu_old *= old_shape[i];
          accu_new *= newshape[i];
        }

        std::vector<cytnx_int64> old_inds(old_shape.size());
        cytnx_int64 j, new_loc = 0;
        for (cytnx_int64 n = 0; n < accu_old; n++) {
          bool recalc = 0;
          for (int i = (int)old_shape.size() - 1; i >= 0; i--) {
            if (old_inds[i] >= old_shape[i]) {
              recalc = 1;
              old_inds[i - 1]++;
              old_inds[i] = 0;
            } else
              break;
          }
          if (recalc) {
            new_loc = 0;
            for (j = 0; j < old_shape.size(); j++) {
              new_loc += shifter_new[j] * old_inds[mapper[j]];
            }
          } else {
            if (n != 0) new_loc += shifter_new[invmapper[old_shape.size() - 1]];
          }
          des[new_loc] = src[n];

          old_inds[old_shape.size() - 1]++;
        }
      }
#else

      std::vector<cytnx_int64> newshape(old_shape.size());
      for (cytnx_int64 i = 0; i < old_shape.size(); i++) {
        newshape[i] = old_shape[mapper[i]];
      }
      std::vector<cytnx_int64> shifter_new(old_shape.size());
      std::vector<cytnx_int64> shifter_old(old_shape.size());

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
      std::vector<cytnx_int64> old_inds(old_shape.size());
      cytnx_int64 j, new_loc = 0;
      for (cytnx_int64 n = 0; n < accu_old; n++) {
        bool recalc = 0;
        for (int i = (int)old_shape.size() - 1; i >= 0; i--) {
          if (old_inds[i] >= old_shape[i]) {
            recalc = 1;
            old_inds[i - 1]++;
            old_inds[i] = 0;
          } else
            break;
        }
        if (recalc) {
          new_loc = 0;
          for (j = 0; j < old_shape.size(); j++) {
            new_loc += shifter_new[j] * old_inds[mapper[j]];
          }
        } else {
          if (n != 0) new_loc += shifter_new[invmapper[old_shape.size() - 1]];
        }
        des[new_loc] = src[n];

        old_inds[old_shape.size() - 1]++;
      }
  #endif
#endif  // hptt
      boost::intrusive_ptr<Storage_base> out;
      if constexpr (std::is_same_v<T, cytnx_complex128>) {
        out = new ComplexDoubleStorage();
      } else if (std::is_same_v<T, cytnx_complex64>) {
        out = new ComplexFloatStorage();
      } else if (std::is_same_v<T, cytnx_double>) {
        out = new DoubleStorage();
      } else if (std::is_same_v<T, cytnx_float>) {
        out = new FloatStorage();
      }
      if (is_inplace) {
        memcpy(in->Mem, des, sizeof(T) * accu_old);
        free(des);
        return out;
      } else {
        out->_Init_byptr(des, accu_old, in->device, true, in->cap);
        return out;
      }
    }

    template boost::intrusive_ptr<Storage_base> MoveMemoryCpu<cytnx_complex128>(
      boost::intrusive_ptr<Storage_base> &, const std::vector<cytnx_uint64> &,
      const std::vector<cytnx_uint64> &, const std::vector<cytnx_uint64> &, const bool is_inplace);
    template boost::intrusive_ptr<Storage_base> MoveMemoryCpu<cytnx_complex64>(
      boost::intrusive_ptr<Storage_base> &, const std::vector<cytnx_uint64> &,
      const std::vector<cytnx_uint64> &, const std::vector<cytnx_uint64> &, const bool is_inplace);
    template boost::intrusive_ptr<Storage_base> MoveMemoryCpu<cytnx_double>(
      boost::intrusive_ptr<Storage_base> &, const std::vector<cytnx_uint64> &,
      const std::vector<cytnx_uint64> &, const std::vector<cytnx_uint64> &, const bool is_inplace);
    template boost::intrusive_ptr<Storage_base> MoveMemoryCpu<cytnx_float>(
      boost::intrusive_ptr<Storage_base> &, const std::vector<cytnx_uint64> &,
      const std::vector<cytnx_uint64> &, const std::vector<cytnx_uint64> &, const bool is_inplace);
    template boost::intrusive_ptr<Storage_base> MoveMemoryCpu<cytnx_uint64>(
      boost::intrusive_ptr<Storage_base> &, const std::vector<cytnx_uint64> &,
      const std::vector<cytnx_uint64> &, const std::vector<cytnx_uint64> &, const bool is_inplace);
    template boost::intrusive_ptr<Storage_base> MoveMemoryCpu<cytnx_int64>(
      boost::intrusive_ptr<Storage_base> &, const std::vector<cytnx_uint64> &,
      const std::vector<cytnx_uint64> &, const std::vector<cytnx_uint64> &, const bool is_inplace);
    template boost::intrusive_ptr<Storage_base> MoveMemoryCpu<cytnx_uint32>(
      boost::intrusive_ptr<Storage_base> &, const std::vector<cytnx_uint64> &,
      const std::vector<cytnx_uint64> &, const std::vector<cytnx_uint64> &, const bool is_inplace);
    template boost::intrusive_ptr<Storage_base> MoveMemoryCpu<cytnx_int32>(
      boost::intrusive_ptr<Storage_base> &, const std::vector<cytnx_uint64> &,
      const std::vector<cytnx_uint64> &, const std::vector<cytnx_uint64> &, const bool is_inplace);
    template boost::intrusive_ptr<Storage_base> MoveMemoryCpu<cytnx_uint16>(
      boost::intrusive_ptr<Storage_base> &, const std::vector<cytnx_uint64> &,
      const std::vector<cytnx_uint64> &, const std::vector<cytnx_uint64> &, const bool is_inplace);
    template boost::intrusive_ptr<Storage_base> MoveMemoryCpu<cytnx_int16>(
      boost::intrusive_ptr<Storage_base> &, const std::vector<cytnx_uint64> &,
      const std::vector<cytnx_uint64> &, const std::vector<cytnx_uint64> &, const bool is_inplace);
    template boost::intrusive_ptr<Storage_base> MoveMemoryCpu<cytnx_bool>(
      boost::intrusive_ptr<Storage_base> &, const std::vector<cytnx_uint64> &,
      const std::vector<cytnx_uint64> &, const std::vector<cytnx_uint64> &, const bool is_inplace);

  }  // namespace utils_internal
}  // namespace cytnx
