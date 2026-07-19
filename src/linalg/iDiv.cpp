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
      detail::check_binary_tensor_inputs(Lt, Rt, "iDiv");
      // A singleton RHS that stays on the host is treated as a broadcast scalar: the GPU kernels
      // read it with a host-side dereference and pass it into the kernel by value, so it needs
      // neither a device match nor a per-call H2D copy of the scalar. See #988.
      const bool rhs_is_scalar = detail::is_singleton_tensor(Rt);
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
      if (Lt.storage().size() == 0) return;

      Tensor R;
      if (Lt._impl->storage()._impl == Rt._impl->storage()._impl) {
        R = Rt.clone();
      } else {
        R = Rt;
      }
      R = detail::host_singleton_for_gpu_broadcast(R, Lt.device());

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
          // #1003: route in-place arithmetic through the typed cuDiv_dispatch instead of the
          // legacy cuAri_ii dtype-pair table. In-place keeps the LHS dtype, so we compute in the
          // promoted dtype and store into Lt's dtype (matching the CPU element-wise semantics).
          // When the promoted dtype already equals the LHS dtype we write straight into Lt's
          // storage, preserving shared-storage aliasing; otherwise we compute into a promoted
          // temporary and narrow back. This also removes the #988 in-place hazards (mapper-
          // ignoring kernels, narrow-LHS OOB writes, host-scalar dereference).
          const unsigned int out_dtype = Type.type_promote(Lt.dtype(), Rt.dtype());
          if (out_dtype == Lt.dtype()) {
            linalg_internal::cuDiv_dispatch(
              Lt._impl->storage()._impl, Lt._impl->storage()._impl, R._impl->storage()._impl,
              Lt._impl->storage()._impl->size(), empty_mapper, empty_mapper, empty_mapper);
          } else {
            Tensor out(Lt.shape(), out_dtype, Lt.device());
            linalg_internal::cuDiv_dispatch(
              out._impl->storage()._impl, Lt._impl->storage()._impl, R._impl->storage()._impl,
              Lt._impl->storage()._impl->size(), empty_mapper, empty_mapper, empty_mapper);
            Lt = out.astype(Lt.dtype());
          }
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
          checkCudaErrors(cudaSetDevice(Lt.device()));
          // #1003: typed dispatch (see the contiguous branch above). A broadcast scalar RHS is
          // layout-uniform, so it is applied in place to Lt's storage in physical order, which
          // preserves a non-contiguous view (and the CPU write-through semantics). For a genuine
          // tensor RHS, cuDiv_dispatch now has a proper non-contiguous kernel that reads both
          // operands through their layout mappers, so the previous "non-contiguous tensor/=tensor
          // is not supported" limitation (the cuAri_ii kernels ignored the mappers, #988) no longer
          // applies; the in-place LHS becomes contiguous.
          const unsigned int out_dtype = Type.type_promote(Lt.dtype(), Rt.dtype());
          if (rhs_is_scalar) {
            if (out_dtype == Lt.dtype()) {
              linalg_internal::cuDiv_dispatch(
                Lt._impl->storage()._impl, Lt._impl->storage()._impl, R._impl->storage()._impl,
                Lt._impl->storage()._impl->size(), empty_mapper, empty_mapper, empty_mapper);
            } else {
              Tensor out = Lt.astype(out_dtype);
              linalg_internal::cuDiv_dispatch(
                out._impl->storage()._impl, out._impl->storage()._impl, R._impl->storage()._impl,
                Lt._impl->storage()._impl->size(), empty_mapper, empty_mapper, empty_mapper);
              Lt = out.astype(Lt.dtype());
            }
          } else {
            Tensor out(Lt.shape(), out_dtype, Lt.device());
            linalg_internal::cuDiv_dispatch(out._impl->storage()._impl, Lt._impl->storage()._impl,
                                            R._impl->storage()._impl,
                                            Lt._impl->storage()._impl->size(), Lt._impl->shape(),
                                            Lt._impl->invmapper(), Rt._impl->invmapper());
            Lt = (out_dtype == Lt.dtype()) ? out : out.astype(Lt.dtype());
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
