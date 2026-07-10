#include "linalg.hpp"

#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"
  #include "Arithmetic_shape.hpp"
  #include "iArithmetic_visit.hpp"

namespace cytnx {
  namespace linalg {

    void iDiv(Tensor &Lt, const Tensor &Rt) {
      // A rank-0 RHS that stays on the host is treated as a broadcast scalar: the GPU kernels read
      // it with a host-side dereference and pass it into the kernel by value, so it needs
      // neither a device match nor a per-call H2D copy of the scalar. See #988.
      const bool rhs_is_scalar = Rt.is_scalar();
      const bool rhs_is_host_scalar = (Rt.device() == Device.cpu && rhs_is_scalar);
      cytnx_error_msg(Lt.device() != Rt.device() && !rhs_is_host_scalar,
                      "[iDiv] The two tensors cannot be on different devices.%s", "\n");
      // In-place ops write the result back into the LHS storage, so a complex result cannot be
      // stored in a real LHS. Guard here (device-independent) so the GPU path throws like the CPU
      // path instead of reinterpreting the real output buffer as complex and corrupting it. See
      // #988.
      cytnx_error_msg(!Type.is_complex(Lt.dtype()) && Type.is_complex(Rt.dtype()),
                      "[iDiv] Cannot perform real /= complex in-place: a complex result cannot be "
                      "stored in a real tensor.%s",
                      "\n");

      if (!rhs_is_scalar) {
        cytnx_error_msg(Lt.shape() != Rt.shape(),
                        "[iDiv] The two tensors do not have the same shape. Lt rank: [%d] "
                        "Rt rank: [%d] %s",
                        Lt.shape().size(), Rt.shape().size(), "\n");
      }

      Tensor R;
      if (Lt._impl->storage()._impl == Rt._impl->storage()._impl) {
        R = Rt.clone();
      } else {
        R = Rt;
      }
      R = detail::host_scalar_for_gpu_broadcast(R, Lt.device());

      // GPU broadcast scalar with a LHS *narrower* than the promoted dtype (e.g. a Float tensor
      // over a Double scalar, or an integer tensor over a fractional scalar): the in-place GPU
      // kernels write the promoted-width result straight into the narrower LHS buffer, an
      // out-of-bounds write that corrupts memory. Compute in the promoted dtype, then truncate
      // back to the LHS dtype -- this matches the CPU element-wise semantics (compute in the
      // promoted type, store into the LHS type). See #988. The CPU path already does this
      // in place, and real /= complex is rejected above.
      if (rhs_is_scalar && Lt.device() != Device.cpu && Lt.dtype() > Rt.dtype()) {
        Tensor promoted = Lt.astype(Type.type_promote(Lt.dtype(), Rt.dtype()));
        iDiv(promoted, R);
        Lt = promoted.astype(Lt.dtype());
        return;
      }

      static const std::vector<cytnx_uint64> empty_mapper;
      // if contiguous, then no need to calculate the mappers
      if ((Lt.is_contiguous() && Rt.is_contiguous())) {
        // contiguous section.
        if (Lt.device() == Device.cpu) {
          detail::DispatchInplaceArithmeticCPU<3>(Lt, R, empty_mapper, empty_mapper, empty_mapper);
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
            Lt._impl->storage()._impl->size(), {}, {}, {}, 3);
          // cytnx_error_msg(true, "[Developing] iAdd for GPU%s", "\n");

          if (Lt.dtype() > Rt.dtype()) Lt = tmpo;

  #else
          cytnx_error_msg(true, "[Div] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
  #endif
        }
      } else {
        // non-contiguous section
        if (Lt.device() == Device.cpu) {
          detail::DispatchInplaceArithmeticCPU<3>(Lt, R, Lt._impl->shape(), Lt._impl->invmapper(),
                                                  Rt._impl->invmapper());
        } else {
  #ifdef UNI_GPU
          if (rhs_is_scalar) {
            // Broadcast scalar RHS: the rconst kernel path divides every element of the
            // (possibly non-contiguous) LHS storage by the scalar in place, so the layout
            // mappers are irrelevant and the result is correct. This is the #988 regression fix
            // that lets e.g. `gpu_tensor.permute(...); gpu_tensor /= 2.0;` succeed again.
            checkCudaErrors(cudaSetDevice(Lt.device()));
            Tensor tmpo;
            if (Lt.dtype() <= Rt.dtype())
              tmpo = Lt;
            else
              tmpo = Lt.clone();
            linalg_internal::lii.cuAri_ii[Lt.dtype()][Rt.dtype()](
              tmpo._impl->storage()._impl, Lt._impl->storage()._impl, R._impl->storage()._impl,
              Lt._impl->storage()._impl->size(), Lt._impl->shape(), Lt._impl->invmapper(),
              Rt._impl->invmapper(), 3);
            if (Lt.dtype() > Rt.dtype()) Lt = tmpo;
          } else {
            // Genuine non-contiguous tensor/=tensor: unlike cuAdd/cuSub, the cuDiv GPU kernels
            // ignore the layout mappers, so routing here would silently pair mismatched
            // elements. Fail loudly instead of corrupting data. Call Contiguous_()/Contiguous()
            // on the operands first.
            cytnx_error_msg(true,
                            "[iDiv][on GPU/CUDA] non-contiguous tensor/=tensor is not supported. "
                            "Call Contiguous_() or Contiguous() on the operands first%s",
                            "\n");
          }
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
