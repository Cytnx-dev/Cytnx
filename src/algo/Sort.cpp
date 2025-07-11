#include "algo.hpp"

#ifdef BACKEND_TORCH
#else

  #include "backend/algo_internal_interface.hpp"
  #include "backend/Storage.hpp"
  #include "backend/Scalar.hpp"
namespace cytnx {
  namespace algo {
    Tensor Sort(const Tensor &Tin) {
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
        cytnx::algo_internal::aii.Sort_ii[out.dtype()](out._impl->storage()._impl,
                                                       out.shape().back(), out.storage().size());

      } else {
  #ifdef UNI_GPU
        cytnx::algo_internal::aii.cuSort_ii[out.dtype()](out._impl->storage()._impl,
                                                         out.shape().back(), out.storage().size());
          // cytnx_error_msg(true, "[Developing] Sort.%s", "\n");
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
