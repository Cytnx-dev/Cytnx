#include "linalg/linalg.hpp"
#include "utils/utils.hpp"
#include "Device.hpp"

namespace cytnx {

  namespace linalg {
    Tensor Otimes(const Tensor &Tl, const Tensor &Tr) {
      // checking:
      cytnx_error_msg(Tl.shape().size() == 0, "[ERROR] pass empty tensor in param #1%s", "\n");
      cytnx_error_msg(Tr.shape().size() == 0, "[ERROR] pass empty tensor in param #2%s", "\n");
      cytnx_error_msg(Tl.device() != Tr.device(),
                      "[ERROR] two tensor cannot on different devices.%s", "\n");
      cytnx_error_msg(!Tl.is_contiguous(),
                      "[ERROR] tensor #1 should be contiguous. suggestion: call "
                      "Tensor.contiguous() or Tensor.contiguous_() first.%s",
                      "\n");
      cytnx_error_msg(!Tr.is_contiguous(),
                      "[ERROR] tensor #2 should be contiguous. suggestion: call "
                      "Tensor.contiguous() or Tensor.contiguous_() first.%s",
                      "\n");

      std::vector<cytnx_uint64> new_shape;
      vec_concatenate_(new_shape, Tl.shape(), Tr.shape());

      Tensor out(new_shape, Tl.dtype() < Tr.dtype() ? Tl.dtype() : Tr.dtype(), Tl.device());

      if (Tl.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Outer_ii[Tl.dtype()][Tr.dtype()](
          out._impl->storage()._impl, Tl._impl->storage()._impl, Tr._impl->storage()._impl);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tl.device()));
        cytnx::linalg_internal::lii.cuOuter_ii[Tl.dtype()][Tr.dtype()](
          out._impl->storage()._impl, Tl._impl->storage()._impl, Tr._impl->storage()._impl);
#else
        cytnx_error_msg(true, "[Otimes] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

  }  // namespace linalg
}  // namespace cytnx
