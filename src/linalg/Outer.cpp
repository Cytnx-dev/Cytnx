#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "utils/utils.hpp"
#include "Device.hpp"
#include "Tensor.hpp"
#include <iostream>
namespace cytnx {

  namespace linalg {
    Tensor Outer(const Tensor &Tl, const Tensor &Tr) {
      // checking:
      cytnx_error_msg(Tl.shape().size() == 0, "[ERROR] pass empty tensor in param #1%s", "\n");
      cytnx_error_msg(Tr.shape().size() == 0, "[ERROR] pass empty tensor in param #2%s", "\n");
      cytnx_error_msg(Tl.device() != Tr.device(),
                      "[ERROR] two tensor cannot on different devices.%s", "\n");
      // cytnx_error_msg(!Tl.is_contiguous(),"[ERROR] tensor #1 should be contiguous. suggestion:
      // call Tensor.contiguous() or Tensor.contiguous_() first.%s","\n");
      // cytnx_error_msg(!Tr.is_contiguous(),"[ERROR] tensor #2 should be contiguous. suggestion:
      // call Tensor.contiguous() or Tensor.contiguous_() first.%s","\n");

      cytnx_error_msg(Tl.shape().size() != 1, "[ERROR] tensor #1 should have rank-1.%s", "\n");
      cytnx_error_msg(Tr.shape().size() != 1, "[ERROR] tensor #2 should have rank-1.%s", "\n");

      //[Note] since here we only have two Tensor with rank-1 so there is no contiguous issue,

      std::vector<cytnx_uint64> new_shape = {Tl.shape()[0], Tr.shape()[0]};

      Tensor out(new_shape, Tl.dtype() < Tr.dtype() ? Tl.dtype() : Tr.dtype(), Tl.device());
      cytnx_uint64 j1, j2;
      j1 = Tl.shape()[0];
      j2 = Tr.shape()[0];

      if (Tl.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Outer_ii[Tl.dtype()][Tr.dtype()](
          out._impl->storage()._impl, Tl._impl->storage()._impl, Tr._impl->storage()._impl, j1, j2);
      } else {
#ifdef UNI_GPU
        cytnx_error_msg(true, "[Outer] currently Outer is not support for GPU, pending for fix.%s",
                        "\n");
        checkCudaErrors(cudaSetDevice(Tl.device()));
        cytnx::linalg_internal::lii.cuOuter_ii[Tl.dtype()][Tr.dtype()](
          out._impl->storage()._impl, Tl._impl->storage()._impl, Tr._impl->storage()._impl, j1, j2);
#else
        cytnx_error_msg(true, "[Outer] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

  }  // namespace linalg
}  // namespace cytnx
