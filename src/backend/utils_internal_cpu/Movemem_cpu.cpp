#include "Movemem_cpu.hpp"

#include <type_traits>
#include <vector>

#include "boost/smart_ptr/intrusive_ptr.hpp"

#include "backend/Storage.hpp"

#ifdef UNI_HPTT
  #include <omp.h>

  #include <algorithm>
  #include <bit>
  #include <cstddef>
  #include <numeric>

  #include "Device.hpp"
  #include "hptt.h"
#endif

namespace cytnx {

  namespace utils_internal {

#ifdef UNI_HPTT
    namespace {

      // Bytes of a transpose per thread when Device.Ncpus allows more than one (its default is 1).
      // Starting a thread team costs microseconds, so a transpose gets one more thread per 64 KiB
      // of data and runs on one thread below 128 KiB. Timed on isolated transposes across sizes and
      // thread counts on a Zen 5 Threadripper (libgomp) and a dual-socket Xeon (libiomp5); see
      // #939.
      constexpr std::size_t kTransposeBytesPerThread = 64 * 1024;

      // Threads for an HPTT transpose of `bytes`: the largest power of two up to
      // bytes / kTransposeBytesPerThread, at most Device.Ncpus, and at most OpenMP's current
      // thread count. HPTT turns the count into an explicit num_threads clause, which would
      // otherwise override OMP_NUM_THREADS and omp_set_num_threads().
      int transpose_threads(std::size_t bytes) {
        const std::size_t by_size =
          std::bit_floor(std::max<std::size_t>(1, bytes / kTransposeBytesPerThread));
        const int cap = std::max(1, std::min(cytnx::Device.Ncpus, omp_get_max_threads()));
        return static_cast<int>(std::min(by_size, static_cast<std::size_t>(cap)));
      }

    }  // namespace
#endif

    template <typename T, typename std::enable_if_t<std::is_integral_v<T>, bool>>
    boost::intrusive_ptr<Storage_base> MoveMemoryCpu(boost::intrusive_ptr<Storage_base> &in,
                                                     const std::vector<cytnx_uint64> &old_shape,
                                                     const std::vector<cytnx_uint64> &mapper,
                                                     const std::vector<cytnx_uint64> &invmapper,
                                                     const bool is_inplace) {
#ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype() != Type.cy_typeid(T()),
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

      T *des = (T *)malloc(in->size() * sizeof(T));
      T *src = static_cast<T *>(in->data());

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
        memcpy(in->data(), des, sizeof(T) * accu_old);
        free(des);
        return out;
      } else {
        out->_Init_byptr(des, accu_old, in->device(), true, in->size());
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
        in->dtype() != Type.cy_typeid(T()),
        "[DEBUG][internal error] in.dtype_str is [%s] but call MoveMemoryCpu with type %s",
        in->dtype_str().c_str(), Type.getname(Type.cy_typeid(T())));
#endif

      T *des = (T *)malloc(in->size() * sizeof(T));
      T *src = static_cast<T *>(in->data());
      cytnx_uint64 accu_old = 1, accu_new = 1;

#ifdef UNI_HPTT
      if (in->size() > 64) {
        std::vector<int> perm(mapper.begin(), mapper.end());
        std::vector<int> size(old_shape.begin(), old_shape.end());
        const int num_threads = transpose_threads(in->size() * sizeof(T));
        // HPTT copies only on a calling thread whose omp_get_thread_num() is in its thread-id list,
        // which defaults to 0 .. num_threads - 1. Inside a user's OpenMP parallel region the caller
        // can have a larger id (any thread but 0 when one thread is allowed there) and would get an
        // uninitialised result, so start the list at the caller's own id.
        std::vector<int> thread_ids(num_threads);
        std::iota(thread_ids.begin(), thread_ids.end(), omp_get_thread_num());
        auto plan =
          hptt::create_plan(&perm[0], perm.size(), 1, src, &size[0], nullptr, 0, des, nullptr,
                            hptt::ESTIMATE, num_threads, thread_ids.data(), true);
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
        memcpy(in->data(), des, sizeof(T) * accu_old);
        free(des);
        return out;
      } else {
        out->_Init_byptr(des, accu_old, in->device(), true, in->size());
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
