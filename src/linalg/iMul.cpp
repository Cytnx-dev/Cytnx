#include "linalg.hpp"

#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"
  #include "Arithmetic_shape.hpp"
  #include "iArithmetic_visit.hpp"
namespace cytnx {
  namespace linalg {

    void iMul(Tensor &Lt, const Tensor &Rt, bool rhs_is_weak_scalar) {
      detail::check_binary_tensor_inputs(Lt, Rt, "iMul");
      // A singleton RHS that stays on the host is treated as a broadcast scalar: the GPU kernels
      // read it with a host-side dereference and pass it into the kernel by value, so it needs
      // neither a device match nor a per-call H2D copy of the scalar. See #988.
      const bool rhs_is_scalar = detail::is_singleton_tensor(Rt);
      const bool rhs_is_host_scalar = (Rt.device() == Device.cpu && rhs_is_scalar);
      cytnx_error_msg(Lt.device() != Rt.device() && !rhs_is_host_scalar,
                      "[iMul] The two tensors cannot be on different devices.%s", "\n");
      // In-place ops write the result back into the LHS storage, so a complex result cannot be
      // stored in a real LHS. Guard here (device-independent) so the GPU path throws like the CPU
      // path instead of reinterpreting the real output buffer as complex and corrupting it. See
      // #988.
      cytnx_error_msg(!Type.is_complex(Lt.dtype()) && Type.is_complex(Rt.dtype()),
                      "[iMul] Cannot perform real *= complex in-place: a complex result cannot be "
                      "stored in a real tensor.%s",
                      "\n");
      if (!rhs_is_scalar) {
        cytnx_error_msg(Lt.shape() != Rt.shape(),
                        "[iMul] The two tensors do not have the same shape. Lt rank: [%d] "
                        "Rt rank: [%d] %s",
                        Lt.shape().size(), Rt.shape().size(), "\n");
      }
      // A zero-extent tensor has nothing to compute, but in-place arithmetic must still
      // promote Lt's dtype to match the non-empty path and the out-of-place operator (#941):
      // the CPU dispatcher below performs that dtype replacement with a no-op kernel when the
      // length is 0. Only short-circuit the legacy GPU path here (GPU keeps the LHS dtype in
      // place regardless; #1013).
      if (Lt.storage().size() == 0 && Lt.device() != Device.cpu) return;

      Tensor R;
      if (Lt._impl->storage()._impl == Rt._impl->storage()._impl) {
        R = Rt.clone();
        // cytnx_warning_msg(true,"[iMul][Warning!] A += B is performed on two object share same
        // memory!%s","\n");
      } else {
        R = Rt;
      }
      R = detail::host_singleton_for_gpu_broadcast(R, Lt.device());

      // GPU broadcast scalar with a LHS *narrower* than the promoted dtype (e.g. a Float tensor
      // times a Double scalar, or an integer tensor times a fractional scalar): the in-place GPU
      // kernels write the promoted-width result straight into the narrower LHS buffer, an
      // out-of-bounds write that corrupts memory. Compute in the promoted dtype, then truncate
      // back to the LHS dtype -- this matches the CPU element-wise semantics (compute in the
      // promoted type, store into the LHS type). See #988. The CPU path already does this
      // in place, and real *= complex is rejected above.
      if (rhs_is_scalar && Lt.device() != Device.cpu && Lt.dtype() > Rt.dtype()) {
        Tensor promoted = Lt.astype(Type.type_promote(Lt.dtype(), Rt.dtype()));
        iMul(promoted, R);
        Lt = promoted.astype(Lt.dtype());
        return;
      }

      static const std::vector<cytnx_uint64> empty_mapper;
      // if contiguous, then no need to calculate the mappers
      if ((Lt.is_contiguous() && Rt.is_contiguous())) {
        // contiguous section.
        if (Lt.device() == Device.cpu) {
          detail::DispatchInplaceArithmeticCPU<1>(Lt, R, rhs_is_weak_scalar, empty_mapper,
                                                  empty_mapper, empty_mapper);
        } else {
  #ifdef UNI_GPU
          checkCudaErrors(cudaSetDevice(Lt.device()));
          Tensor tmpo;
          if (Lt.dtype() <= Rt.dtype())
            tmpo = Lt;
          else
            tmpo = Lt.clone();
          linalg_internal::lii.cuAri_ii[Lt.dtype()][Rt.dtype()](
            tmpo._impl->storage()._impl, Lt._impl->storage()._impl, R._impl->storage()._impl,
            Lt._impl->storage()._impl->size(), {}, {}, {}, 1);
          // cytnx_error_msg(true, "[Developing] iAdd for GPU%s", "\n");

          if (Lt.dtype() > Rt.dtype()) Lt = tmpo;

  #else
          cytnx_error_msg(true, "[Mul] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
  #endif
        }
      } else {
        // non-contiguous section
        if (Lt.device() == Device.cpu) {
          detail::DispatchInplaceArithmeticCPU<1>(Lt, R, rhs_is_weak_scalar, Lt._impl->shape(),
                                                  Lt._impl->invmapper(), Rt._impl->invmapper());
        } else {
  #ifdef UNI_GPU
          if (rhs_is_scalar) {
            // Broadcast scalar RHS: the rconst kernel path scales every element of the
            // (possibly non-contiguous) LHS storage in place, so the layout mappers are
            // irrelevant and the result is correct. This is the #988 regression fix that lets
            // e.g. `gpu_tensor.permute(...); gpu_tensor *= 2.0;` succeed again.
            checkCudaErrors(cudaSetDevice(Lt.device()));
            Tensor tmpo;
            if (Lt.dtype() <= Rt.dtype())
              tmpo = Lt;
            else
              tmpo = Lt.clone();
            linalg_internal::lii.cuAri_ii[Lt.dtype()][Rt.dtype()](
              tmpo._impl->storage()._impl, Lt._impl->storage()._impl, R._impl->storage()._impl,
              Lt._impl->storage()._impl->size(), Lt._impl->shape(), Lt._impl->invmapper(),
              Rt._impl->invmapper(), 1);
            if (Lt.dtype() > Rt.dtype()) Lt = tmpo;
          } else {
            // Genuine non-contiguous tensor*=tensor: unlike cuAdd/cuSub, the cuMul GPU kernels
            // ignore the layout mappers, so routing here would silently pair mismatched
            // elements. Fail loudly instead of corrupting data. Call Contiguous_()/Contiguous()
            // on the operands first.
            cytnx_error_msg(true,
                            "[iMul][on GPU/CUDA] non-contiguous tensor*=tensor is not supported. "
                            "Call Contiguous_() or Contiguous() on the operands first%s",
                            "\n");
          }
  #else
          cytnx_error_msg(true, "[Mul] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
  #endif
        }
      }
    }

  }  // namespace linalg
}  // namespace cytnx
#endif  // BACKEND_TORCH
