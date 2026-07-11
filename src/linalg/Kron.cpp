#include "linalg.hpp"
#include "utils/utils.hpp"
#include "Device.hpp"
#include <iostream>
#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else

  #include "backend/linalg_internal_cpu/Kron_internal.hpp"

  #ifdef UNI_GPU
    #include "backend/linalg_internal_gpu/cuKron_internal.cuh"
  #endif

namespace cytnx {

  namespace linalg {
    Tensor Kron(const Tensor &lhs, const Tensor &rhs, bool lhs_pad_left, bool rhs_pad_left) {
      // checking:
      cytnx_error_msg(lhs.device() != rhs.device(),
                      "[ERROR] two tensor cannot on different devices.%s", "\n");
      cytnx_error_msg(lhs.is_void(), "[ERROR] pass empty tensor in param #1%s", "\n");
      cytnx_error_msg(rhs.is_void(), "[ERROR] pass empty tensor in param #2%s", "\n");
      if (lhs.is_scalar() || rhs.is_scalar()) return lhs * rhs;
      // cytnx_error_msg(!Tl.is_contiguous(),"[ERROR] tensor #1 should be contiguous. suggestion:
      // call Tensor.contiguous() or Tensor.contiguous_() first.%s","\n");
      // cytnx_error_msg(!Tr.is_contiguous(),"[ERROR] tensor #2 should be contiguous. suggestion:
      // call Tensor.contiguous() or Tensor.contiguous_() first.%s","\n");

      auto lhs_contiguous = lhs.contiguous();
      auto rhs_contiguous = rhs.contiguous();

      // check the new shape:

      std::vector<cytnx_uint64> new_shape;
      std::vector<cytnx_uint64> lhs_padded_shape = lhs_contiguous.shape();
      std::vector<cytnx_uint64> rhs_padded_shape = rhs_contiguous.shape();
      const auto lhs_rank = lhs_padded_shape.size();
      const auto rhs_rank = rhs_padded_shape.size();
      std::vector<cytnx_uint64> ones(
        lhs_rank > rhs_rank ? lhs_rank - rhs_rank : rhs_rank - lhs_rank, 1);
      if (lhs_rank > rhs_rank) {
        if (!rhs_pad_left) {
          rhs_padded_shape.insert(rhs_padded_shape.end(), std::make_move_iterator(ones.begin()),
                                  std::make_move_iterator(ones.end()));
        } else {
          ones.insert(ones.end(), std::make_move_iterator(rhs_padded_shape.begin()),
                      std::make_move_iterator(rhs_padded_shape.end()));
          rhs_padded_shape = std::move(ones);
        }
      } else if (lhs_rank < rhs_rank) {
        if (!lhs_pad_left) {
          lhs_padded_shape.insert(lhs_padded_shape.end(), std::make_move_iterator(ones.begin()),
                                  std::make_move_iterator(ones.end()));
        } else {
          ones.insert(ones.end(), std::make_move_iterator(lhs_padded_shape.begin()),
                      std::make_move_iterator(lhs_padded_shape.end()));
          lhs_padded_shape = std::move(ones);
        }
      }
      new_shape.resize(lhs_padded_shape.size());

      for (unsigned long long i = 0; i < new_shape.size(); i++) {
        new_shape[i] = lhs_padded_shape[i] * rhs_padded_shape[i];
      }

      Tensor out(new_shape, Type.type_promote(lhs_contiguous.dtype(), rhs_contiguous.dtype()),
                 lhs_contiguous.device());
      if (out.is_empty()) return out;

      if (lhs_contiguous.device() == Device.cpu) {
        // Dispatch to the kernel based on the types of lhs and rhs.
        std::visit(
          [&](auto lhs_ptr, auto rhs_ptr) {
            using out_type =
              Type_class::type_promote_from_pointer_t<decltype(lhs_ptr), decltype(rhs_ptr)>;
            static_assert(!std::is_same_v<out_type, void>);
            cytnx::linalg_internal::Kron_general(out.ptr_as<out_type>(), lhs_ptr, rhs_ptr,
                                                 lhs_padded_shape, rhs_padded_shape);
          },
          lhs_contiguous.ptr(), rhs_contiguous.ptr());

      } else {
  #ifdef UNI_GPU
        // checkCudaErrors(cudaSetDevice(lhs_contiguous.device()));
        // Dispatch to the kernel based on the types of lhs and rhs.
        std::visit(
          [&](auto lhs_ptr, auto rhs_ptr) {
            using out_type =
              Type_class::type_promote_from_gpu_pointer_t<decltype(lhs_ptr), decltype(rhs_ptr)>;
            static_assert(!std::is_same_v<out_type, void>);
            cytnx::linalg_internal::cuKron_general(out.gpu_ptr_as<out_type>(), lhs_ptr, rhs_ptr,
                                                   lhs_padded_shape, rhs_padded_shape);
          },
          lhs_contiguous.gpu_ptr(), rhs_contiguous.gpu_ptr());
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
