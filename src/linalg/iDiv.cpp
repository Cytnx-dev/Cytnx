#include "linalg.hpp"

#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"
  #include "Arithmetic_shape.hpp"
  #include "iArithmetic_visit.hpp"
  #ifdef UNI_GPU
    #include "backend/linalg_internal_gpu/cuiArithmetic_dispatch.hpp"
  #endif

namespace cytnx {
  namespace linalg {

    void iDiv(Tensor &Lt, const Tensor &Rt, bool rhs_is_weak_scalar) {
      detail::check_binary_tensor_inputs(Lt, Rt, "iDiv");
      // A singleton RHS that stays on the host is treated as a broadcast scalar: the GPU kernels
      // read it with a host-side dereference and pass it into the kernel by value, so it needs
      // neither a device match nor a per-call H2D copy of the scalar. See #988.
      const bool rhs_is_scalar = detail::is_singleton_tensor(Rt);
      const bool rhs_is_host_scalar = (Rt.device() == Device.cpu && rhs_is_scalar);
      cytnx_error_msg(Lt.device() != Rt.device() && !rhs_is_host_scalar,
                      "[iDiv] The two tensors cannot be on different devices.%s", "\n");
      // Reject only a complex python *weak scalar* into a real LHS. numpy weak-scalar
      // semantics (#980/#1015) keep the LHS dtype, so a complex weak scalar cannot be
      // stored in a real tensor. A GENUINE complex tensor RHS is allowed: it promotes
      // Lt's storage to complex like the out-of-place op (#941/#1013). This guard stays
      // device-independent because the GPU kernel's complex-into-real branch silently
      // returns zero instead of throwing (see cuiArithmeticDispatch.cuh) and so relies
      // on this host-side rejection. See #988.
      cytnx_error_msg(
        rhs_is_weak_scalar && !Type.is_complex(Lt.dtype()) && Type.is_complex(Rt.dtype()),
        "[iDiv] Cannot perform real /= complex-scalar in-place: weak-scalar "
        "semantics preserve the real LHS dtype, so the complex scalar cannot be "
        "stored. Use a complex LHS, or a genuine complex tensor RHS to promote.%s",
        "\n");

      if (!rhs_is_scalar) {
        cytnx_error_msg(Lt.shape() != Rt.shape(),
                        "[iDiv] The two tensors do not have the same shape. Lt rank: [%d] "
                        "Rt rank: [%d] %s",
                        Lt.shape().size(), Rt.shape().size(), "\n");
      }
      // A zero-extent tensor has nothing to compute, but in-place arithmetic must still
      // promote Lt's dtype to match the non-empty path and the out-of-place operator (#941).
      // Both the CPU and (as of #1013) the GPU dispatchers below perform that dtype
      // replacement with a no-op kernel when the length is 0, so no early return is needed.
      Tensor R;
      if (Lt._impl->storage()._impl == Rt._impl->storage()._impl) {
        R = Rt.clone();
      } else {
        R = Rt;
      }
      R = detail::host_singleton_for_gpu_broadcast(R, Lt.device());

      // The GPU dispatch below promotes Lt's storage to the output dtype like the CPU
      // path (#1013) -- for Div that is true division (make_floating_point of the
      // promoted type) -- so a genuine higher-precision scalar RHS no longer needs the
      // legacy promote-then-truncate dance (which existed to avoid the narrow-LHS
      // out-of-bounds write of #988); a python weak-scalar RHS still preserves the LHS
      // dtype (its floating form for Div) via rhs_is_weak_scalar. real /= complex-scalar
      // is rejected above; a genuine complex tensor RHS promotes the real LHS to complex.
      static const std::vector<cytnx_uint64> empty_mapper;
      // if contiguous, then no need to calculate the mappers
      if ((Lt.is_contiguous() && Rt.is_contiguous())) {
        // contiguous section.
        if (Lt.device() == Device.cpu) {
          detail::DispatchInplaceArithmeticCPU<3>(Lt, R, rhs_is_weak_scalar, empty_mapper,
                                                  empty_mapper, empty_mapper);
        } else {
  #ifdef UNI_GPU
          linalg_internal::cuiArithmeticDispatch(3, Lt, R, rhs_is_weak_scalar, empty_mapper,
                                                 empty_mapper, empty_mapper);
  #else
          cytnx_error_msg(true, "[Div] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
  #endif
        }
      } else {
        // non-contiguous section
        if (Lt.device() == Device.cpu) {
          detail::DispatchInplaceArithmeticCPU<3>(Lt, R, rhs_is_weak_scalar, Lt._impl->shape(),
                                                  Lt._impl->invmapper(), Rt._impl->invmapper());
        } else {
  #ifdef UNI_GPU
          // #1013: the typed in-place dispatch's non-contiguous kernel applies the layout
          // mappers correctly, so non-contiguous tensor /= tensor -- previously rejected
          // because the legacy cuDiv kernels ignored the mappers (#988) -- is now
          // supported in place like Add/Sub.
          linalg_internal::cuiArithmeticDispatch(3, Lt, R, rhs_is_weak_scalar, Lt._impl->shape(),
                                                 Lt._impl->invmapper(), Rt._impl->invmapper());
  #else
          cytnx_error_msg(true, "[Div] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
  #endif
        }
      }
    }

  }  // namespace linalg
}  // namespace cytnx
#endif  // BACKEND_TORCH
