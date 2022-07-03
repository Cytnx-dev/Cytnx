#include "algo.hpp"
#include "algo_internal_interface.hpp"

namespace cytnx {
  namespace algo {
    Tensor Sort(const Tensor &Tin) {
      Tensor out;
      if (Tin.is_contiguous())
        out = Tin.clone();
      else
        out = Tin.contiguous();

      if (Tin.device() == Device.cpu) {
        cytnx::algo_internal::aii.Sort_ii[out.dtype()](out._impl->storage()._impl,
                                                       out.shape().back(), out.storage().size());

      } else {
#ifdef UNI_GPU
        cytnx_error_msg(true, "[Developing] Sort.%s", "\n");
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
