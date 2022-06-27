#include "linalg.hpp"
#include "utils/utils.hpp"
#include "Device.hpp"
#include <iostream>
#include "linalg_internal_interface.hpp"
#include "Tensor.hpp"
namespace cytnx {

  namespace linalg {
    Tensor Kron(const Tensor &_Tl, const Tensor &_Tr, const bool &Tl_pad_left,
                const bool &Tr_pad_left) {
      // checking:
      cytnx_error_msg(_Tl.shape().size() == 0, "[ERROR] pass empty tensor in param #1%s", "\n");
      cytnx_error_msg(_Tr.shape().size() == 0, "[ERROR] pass empty tensor in param #2%s", "\n");
      cytnx_error_msg(_Tl.device() != _Tr.device(),
                      "[ERROR] two tensor cannot on different devices.%s", "\n");
      // cytnx_error_msg(!Tl.is_contiguous(),"[ERROR] tensor #1 should be contiguous. suggestion:
      // call Tensor.contiguous() or Tensor.contiguous_() first.%s","\n");
      // cytnx_error_msg(!Tr.is_contiguous(),"[ERROR] tensor #2 should be contiguous. suggestion:
      // call Tensor.contiguous() or Tensor.contiguous_() first.%s","\n");

      auto Tl = _Tl.contiguous();
      auto Tr = _Tr.contiguous();

      // check the new shape:

      std::vector<cytnx_uint64> new_shape;
      std::vector<cytnx_uint64> pad_shape1 = Tl.shape();
      std::vector<cytnx_uint64> pad_shape2 = Tr.shape();
      std::vector<cytnx_uint64> ones(
        std::abs((long long)Tl.shape().size() - (long long)Tr.shape().size()), 1);
      if (Tl.shape().size() > Tr.shape().size()) {
        if (Tr_pad_left == false) {
          pad_shape2.insert(pad_shape2.end(), std::make_move_iterator(ones.begin()),
                            std::make_move_iterator(ones.end()));
        } else {
          ones.insert(ones.end(), std::make_move_iterator(pad_shape2.begin()),
                      std::make_move_iterator(pad_shape2.end()));
          pad_shape2 = std::move(ones);
        }
      } else if (Tl.shape().size() < Tr.shape().size()) {
        if (Tl_pad_left == false) {
          pad_shape1.insert(pad_shape1.end(), std::make_move_iterator(ones.begin()),
                            std::make_move_iterator(ones.end()));
        } else {
          ones.insert(ones.end(), std::make_move_iterator(pad_shape2.begin()),
                      std::make_move_iterator(pad_shape2.end()));
          pad_shape1 = std::move(ones);
        }
      }
      new_shape.resize(pad_shape1.size());

      for (unsigned long long i = 0; i < new_shape.size(); i++) {
        new_shape[i] = pad_shape1[i] * pad_shape2[i];
      }

      Tensor out(new_shape, Tl.dtype() < Tr.dtype() ? Tl.dtype() : Tr.dtype(), Tl.device());

      if (Tl.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Kron_ii[Tl.dtype()][Tr.dtype()](
          out._impl->storage()._impl, Tl._impl->storage()._impl, Tr._impl->storage()._impl,
          pad_shape1, pad_shape2);
      } else {
#ifdef UNI_GPU
        cytnx_error_msg(true, "[Kron] currently Kron is not support for GPU, pending for fix.%s",
                        "\n");
        checkCudaErrors(cudaSetDevice(Tl.device()));
        // cytnx::linalg_internal::lii.cuOuter_ii[Tl.dtype()][Tr.dtype()](out._impl->storage()._impl,
        // Tl._impl->storage()._impl, Tr._impl->storage()._impl,i1,j1,i2,j2);
#else
        cytnx_error_msg(true, "[Kron] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

  }  // namespace linalg
}  // namespace cytnx
