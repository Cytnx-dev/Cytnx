#include "linalg.hpp"
#include "utils/utils.hpp"
#include "Device.hpp"
#include <iostream>
#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else

  #include "../backend/linalg_internal_cpu/Kron_internal.hpp"

  #ifdef UNI_GPU
    #include "../backend/linalg_internal_gpu/cuKron_internal.cuh"
  #endif

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

      Tensor out(new_shape, Type.type_promote(Tl.dtype(), Tr.dtype()), Tl.device());

      if (Tl.device() == Device.cpu) {
        // Dispatch to the kernel based on the types of Tl and Tr
        std::visit(
          [&](auto tl, auto tr) {
            // tl and tr are pointer types here.
            using out_type = Type_class::type_promote_from_pointer_t<decltype(tl), decltype(tr)>;
            static_assert(!std::is_same_v<out_type, void>);
            cytnx::linalg_internal::Kron_general(out.ptr_as<out_type>(), tl, tr, pad_shape1,
                                                 pad_shape2);
          },
          Tl.ptr(), Tr.ptr());

      } else {
  #ifdef UNI_GPU
        // checkCudaErrors(cudaSetDevice(Tl.device()));
        // Dispatch to the kernel based on the types of Tl and Tr
        std::visit(
          [&](auto tl, auto tr) {
            // tl and tr are pointer types here.
            using out_type =
              Type_class::type_promote_from_gpu_pointer_t<decltype(tl), decltype(tr)>;
            static_assert(!std::is_same_v<out_type, void>);
            cytnx::linalg_internal::cuKron_general(out.ptr_as<out_type>(), tl, tr, pad_shape1,
                                                   pad_shape2);
          },
          Tl.gpu_ptr(), Tr.gpu_ptr());
  #else
        cytnx_error_msg(true, "[Kron] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

  }  // namespace linalg
}  // namespace cytnx
#endif  // BACKEND_TORCH
