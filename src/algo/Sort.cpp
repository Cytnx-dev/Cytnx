#include "algo.hpp"

#ifdef BACKEND_TORCH
#else

  #include "backend/algo_internal_cpu/Sort_internal.hpp"
  #ifdef UNI_GPU
    #include "backend/algo_internal_gpu/cuSort_internal.cuh"
  #endif
  #include "backend/Storage.hpp"
  #include "backend/Scalar.hpp"
namespace cytnx {
  namespace algo {
    Tensor Sort(const Tensor& Tin) {
      Tensor out;
      if (Tin.is_contiguous())
        out = Tin.clone();
      else
        out = Tin.contiguous();

      // Handle edge case: if last dimension is 0, there's nothing to sort
      // Note: This branch may never be reached as exception is thrown when creating tensor with 0
      // dim
      if (Tin.shape().back() == 0) return out;

      if (Tin.device() == Device.cpu) {
        std::visit(
          [&](auto ptr) {
            using out_type = std::remove_pointer_t<decltype(ptr)>;
            static_assert(!std::is_same_v<out_type, void>);
            cytnx::algo_internal::SortInternalImpl<out_type>(
              out._impl->storage()._impl, out.shape().back(), out.storage().size());
          },
          out.ptr());

      } else {
  #ifdef UNI_GPU
        std::visit(
          [&](auto ptr) {
            using out_type = std::remove_pointer_t<decltype(ptr)>;
            static_assert(!std::is_same_v<out_type, void>);
            cytnx::algo_internal::cuSortInternalImpl<out_type>(
              out._impl->storage()._impl, out.shape().back(), out.storage().size());
          },
          out.gpu_ptr());
  #else
        cytnx_error_msg(true, "[Svd] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
          // return std::vector<Tensor>();
  #endif
      }
      return out;
    }
  }  // namespace algo
}  // namespace cytnx

#endif  // BACKEND_TORCH
