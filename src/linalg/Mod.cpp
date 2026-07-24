#include "linalg.hpp"

#include "Tensor.hpp"
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "Arithmetic_shape.hpp"
  #include "backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {
    Tensor Mod(const Tensor &Lt, const Tensor &Rt) {
      detail::check_binary_tensor_inputs(Lt, Rt, "Mod");
      cytnx_error_msg(Lt.device() != Rt.device(),
                      "[Mod] The two tensors cannot be on different devices.%s", "\n");
      if ((Lt.is_scalar() || Rt.is_scalar()) && Lt.device() != Device.cpu) {
        return Mod(Lt.to(Device.cpu), Rt.to(Device.cpu)).to(Lt.device());
      }

      Tensor out;
      bool icnst = false;
      // The Mod kernels assign every output element (_out[i] = ...), so out is
      // fully overwritten -- pass false to skip the redundant zero-initialization.
      if (detail::init_broadcast_binary_output(out, Lt, Rt,
                                               Type.type_promote(Lt.dtype(), Rt.dtype()), false)) {
        icnst = true;
      } else {
        cytnx_error_msg(Lt.shape() != Rt.shape(),
                        "[Mod] The two tensors do not have the same shape.%s", "\n");
        out.Init(Lt.shape(), Type.type_promote(Lt.dtype(), Rt.dtype()), Lt.device(), false);
      }

      const Tensor left = detail::host_singleton_for_gpu_broadcast(Lt, Lt.device());
      const Tensor right = detail::host_singleton_for_gpu_broadcast(Rt, Lt.device());

      if ((Lt.is_contiguous() && Rt.is_contiguous()) || icnst) {
        // contiguous section
        if (Lt.device() == Device.cpu) {
          std::visit(
            [&](auto lhs_impl, auto rhs_impl) {
              using TL = storage_value_t<decltype(lhs_impl)>;
              using TR = storage_value_t<decltype(rhs_impl)>;
              using TO = Type_class::type_promote_t<TL, TR>;
              cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
                storage_cast<TO>(out._impl->storage()._impl), lhs_impl, rhs_impl,
                out._impl->storage().size(), {}, {}, {});
            },
            Lt._impl->storage().as_storage_variant(), Rt._impl->storage().as_storage_variant());
        } else {
  #ifdef UNI_GPU
          checkCudaErrors(cudaSetDevice(Rt.device()));
          cytnx::linalg_internal::cuMod_dispatch(
            out._impl->storage()._impl, left._impl->storage()._impl, right._impl->storage()._impl,
            out._impl->storage()._impl->size(), {}, {}, {});
  #else
          cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
  #endif
        }
      } else {
        // non-contiguous section
        if (Lt.device() == Device.cpu) {
          std::visit(
            [&](auto lhs_impl, auto rhs_impl) {
              using TL = storage_value_t<decltype(lhs_impl)>;
              using TR = storage_value_t<decltype(rhs_impl)>;
              using TO = Type_class::type_promote_t<TL, TR>;
              cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
                storage_cast<TO>(out._impl->storage()._impl), lhs_impl, rhs_impl,
                out._impl->storage().size(), Lt._impl->shape(), Lt._impl->invmapper(),
                Rt._impl->invmapper());
            },
            Lt._impl->storage().as_storage_variant(), Rt._impl->storage().as_storage_variant());
        } else {
  #ifdef UNI_GPU
          cytnx_error_msg(true,
                          "[Mod][on GPU/CUDA] error two tensors must be contiguous. Call "
                          "Contiguous_() or Contiguous() first%s",
                          "\n");
  #else
          cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
  #endif
        }
      }
      return out;
    }

    //-----------------------------------------------------------------------------------
    template <>
    Tensor Mod<cytnx_complex128>(const cytnx_complex128 &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Mod");
      Storage Cnst(1, Type.ComplexDouble);
      Cnst.at<cytnx_complex128>(0) = lc;

      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();  //(Rt.shape(),Type.ComplexDouble,Rt.device());
      out._impl->storage() = Storage(Rt._impl->storage().size(), Type.ComplexDouble, Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_complex128;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl, Cnst._impl,
                                               Rt._impl->storage()._impl,
                                               Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Mod<cytnx_complex64>(const cytnx_complex64 &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Mod");
      Storage Cnst(1, Type.ComplexFloat);
      Cnst.at<cytnx_complex64>(0) = lc;

      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();  //(Rt.shape(),Type.ComplexDouble,Rt.device());
      out._impl->storage() = Storage(Rt._impl->storage().size(),
                                     Type.type_promote(Type.ComplexFloat, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(),Type.ComplexFloat <
      // Rt.dtype()?Type.ComplexFloat:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_complex64;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl, Cnst._impl,
                                               Rt._impl->storage()._impl,
                                               Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Mod<cytnx_double>(const cytnx_double &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Mod");
      Storage Cnst(1, Type.Double);
      Cnst.at<cytnx_double>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();  //(Rt.shape(),Type.ComplexDouble,Rt.device());
      out._impl->storage() = Storage(Rt._impl->storage().size(),
                                     Type.type_promote(Type.Double, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(),Type.Double < Rt.dtype()?Type.Double:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_double;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl, Cnst._impl,
                                               Rt._impl->storage()._impl,
                                               Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Mod<cytnx_float>(const cytnx_float &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Mod");
      Storage Cnst(1, Type.Float);
      Cnst.at<cytnx_float>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();  //(Rt.shape(),Type.ComplexDouble,Rt.device());
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.type_promote(Type.Float, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(),Type.Float < Rt.dtype()?Type.Float:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_float;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl, Cnst._impl,
                                               Rt._impl->storage()._impl,
                                               Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Mod<cytnx_int64>(const cytnx_int64 &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Mod");
      Storage Cnst(1, Type.Int64);
      Cnst.at<cytnx_int64>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();  //(Rt.shape(),Type.ComplexDouble,Rt.device());
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.type_promote(Type.Int64, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(),Type.Int64 < Rt.dtype()?Type.Int64:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_int64;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl, Cnst._impl,
                                               Rt._impl->storage()._impl,
                                               Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Mod<cytnx_uint64>(const cytnx_uint64 &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Mod");
      Storage Cnst(1, Type.Uint64);
      Cnst.at<cytnx_uint64>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();  //(Rt.shape(),Type.ComplexDouble,Rt.device());
      out._impl->storage() = Storage(Rt._impl->storage().size(),
                                     Type.type_promote(Type.Uint64, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(),Type.Uint64 < Rt.dtype()?Type.Uint64:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_uint64;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl, Cnst._impl,
                                               Rt._impl->storage()._impl,
                                               Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Mod<cytnx_int32>(const cytnx_int32 &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Mod");
      Storage Cnst(1, Type.Int32);
      Cnst.at<cytnx_int32>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();  //(Rt.shape(),Type.ComplexDouble,Rt.device());
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.type_promote(Type.Int32, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(),Type.Int32 < Rt.dtype()?Type.Int32:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_int32;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl, Cnst._impl,
                                               Rt._impl->storage()._impl,
                                               Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Mod<cytnx_uint32>(const cytnx_uint32 &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Mod");
      Storage Cnst(1, Type.Uint32);
      Cnst.at<cytnx_uint32>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();  //(Rt.shape(),Type.ComplexDouble,Rt.device());
      out._impl->storage() = Storage(Rt._impl->storage().size(),
                                     Type.type_promote(Type.Uint32, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(),Type.Uint32 < Rt.dtype()?Type.Uint32:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_uint32;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl, Cnst._impl,
                                               Rt._impl->storage()._impl,
                                               Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Mod<cytnx_int16>(const cytnx_int16 &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Mod");
      Storage Cnst(1, Type.Int16);
      Cnst.at<cytnx_int16>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();  //(Rt.shape(),Type.ComplexDouble,Rt.device());
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.type_promote(Type.Int16, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(),Type.Int16 < Rt.dtype()?Type.Int16:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_int16;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl, Cnst._impl,
                                               Rt._impl->storage()._impl,
                                               Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Mod<cytnx_uint16>(const cytnx_uint16 &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Mod");
      Storage Cnst(1, Type.Uint16);
      Cnst.at<cytnx_uint16>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();  //(Rt.shape(),Type.ComplexDouble,Rt.device());
      out._impl->storage() = Storage(Rt._impl->storage().size(),
                                     Type.type_promote(Type.Uint16, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(),Type.Uint16 < Rt.dtype()?Type.Uint16:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_uint16;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl, Cnst._impl,
                                               Rt._impl->storage()._impl,
                                               Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Mod<cytnx_bool>(const cytnx_bool &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Mod");
      Storage Cnst(1, Type.Bool);
      Cnst.at<cytnx_bool>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();  //(Rt.shape(),Type.ComplexDouble,Rt.device());
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.type_promote(Type.Bool, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(),Type.Bool < Rt.dtype()?Type.Bool:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_bool;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl, Cnst._impl,
                                               Rt._impl->storage()._impl,
                                               Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Mod<Scalar>(const Scalar &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Mod");
      Storage Cnst(1, lc.dtype());
      Cnst.set_item(0, lc);

      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();  //(Rt.shape(),Type.ComplexDouble,Rt.device());
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.type_promote(lc.dtype(), Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(),Type.Bool < Rt.dtype()?Type.Bool:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl, auto rhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Cnst.as_storage_variant(), Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl, Cnst._impl,
                                               Rt._impl->storage()._impl,
                                               Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    //-----------------------------------------------------------------------------------
    template <>
    Tensor Mod<cytnx_complex128>(const Tensor &Lt, const cytnx_complex128 &rc) {
      detail::check_tensor_initialized(Lt, "Mod");
      Storage Cnst(1, Type.ComplexDouble);
      Cnst.at<cytnx_complex128>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Lt._impl->storage().size(), Type.ComplexDouble, Lt.device());
      // Tensor out(Lt.shape(),Type.ComplexDouble,Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_complex128;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl,
                                               Lt._impl->storage()._impl, Cnst._impl,
                                               Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }
    template <>
    Tensor Mod<cytnx_complex64>(const Tensor &Lt, const cytnx_complex64 &rc) {
      detail::check_tensor_initialized(Lt, "Mod");
      Storage Cnst(1, Type.ComplexFloat);
      Cnst.at<cytnx_complex64>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Lt._impl->storage().size(),
                                     Type.type_promote(Type.ComplexFloat, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(),Type.ComplexFloat <
      // Lt.dtype()?Type.ComplexFloat:Lt.dtype(),Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_complex64;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl,
                                               Lt._impl->storage()._impl, Cnst._impl,
                                               Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }
    template <>
    Tensor Mod<cytnx_double>(const Tensor &Lt, const cytnx_double &rc) {
      detail::check_tensor_initialized(Lt, "Mod");
      Storage Cnst(1, Type.Double);
      Cnst.at<cytnx_double>(0) = rc;

      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();  //(Rt.shape(),Type.ComplexDouble,Rt.device());
      out._impl->storage() = Storage(Lt._impl->storage().size(),
                                     Type.type_promote(Type.Double, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(),Type.Double < Lt.dtype()?Type.Double:Lt.dtype(),Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_double;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl,
                                               Lt._impl->storage()._impl, Cnst._impl,
                                               Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }
    template <>
    Tensor Mod<cytnx_float>(const Tensor &Lt, const cytnx_float &rc) {
      detail::check_tensor_initialized(Lt, "Mod");
      Storage Cnst(1, Type.Float);
      Cnst.at<cytnx_float>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Lt._impl->storage().size(), Type.type_promote(Type.Float, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(),Type.Float < Lt.dtype()?Type.Float:Lt.dtype(),Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_float;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl,
                                               Lt._impl->storage()._impl, Cnst._impl,
                                               Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }
    template <>
    Tensor Mod<cytnx_int64>(const Tensor &Lt, const cytnx_int64 &rc) {
      detail::check_tensor_initialized(Lt, "Mod");
      Storage Cnst(1, Type.Int64);
      Cnst.at<cytnx_int64>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Lt._impl->storage().size(), Type.type_promote(Type.Int64, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(),Type.Int64 < Lt.dtype()?Type.Int64:Lt.dtype(),Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_int64;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl,
                                               Lt._impl->storage()._impl, Cnst._impl,
                                               Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }
    template <>
    Tensor Mod<cytnx_uint64>(const Tensor &Lt, const cytnx_uint64 &rc) {
      detail::check_tensor_initialized(Lt, "Mod");
      Storage Cnst(1, Type.Uint64);
      Cnst.at<cytnx_uint64>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Lt._impl->storage().size(),
                                     Type.type_promote(Type.Uint64, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(),Type.Uint64 < Lt.dtype()?Type.Uint64:Lt.dtype(),Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_uint64;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl,
                                               Lt._impl->storage()._impl, Cnst._impl,
                                               Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }
    template <>
    Tensor Mod<cytnx_int32>(const Tensor &Lt, const cytnx_int32 &rc) {
      detail::check_tensor_initialized(Lt, "Mod");
      Storage Cnst(1, Type.Int32);
      Cnst.at<cytnx_int32>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Lt._impl->storage().size(), Type.type_promote(Type.Int32, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(),Type.Int32 < Lt.dtype()?Type.Int32:Lt.dtype(),Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_int32;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl,
                                               Lt._impl->storage()._impl, Cnst._impl,
                                               Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }
    template <>
    Tensor Mod<cytnx_uint32>(const Tensor &Lt, const cytnx_uint32 &rc) {
      detail::check_tensor_initialized(Lt, "Mod");
      Storage Cnst(1, Type.Uint32);
      Cnst.at<cytnx_uint32>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Lt._impl->storage().size(),
                                     Type.type_promote(Type.Uint32, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(),Type.Uint32 < Lt.dtype()?Type.Uint32:Lt.dtype(),Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_uint32;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl,
                                               Lt._impl->storage()._impl, Cnst._impl,
                                               Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Mod<cytnx_int16>(const Tensor &Lt, const cytnx_int16 &rc) {
      detail::check_tensor_initialized(Lt, "Mod");
      Storage Cnst(1, Type.Int16);
      Cnst.at<cytnx_int16>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Lt._impl->storage().size(), Type.type_promote(Type.Int16, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(),Type.Int16 < Lt.dtype()?Type.Int16:Lt.dtype(),Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_int16;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl,
                                               Lt._impl->storage()._impl, Cnst._impl,
                                               Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }
    template <>
    Tensor Mod<cytnx_uint16>(const Tensor &Lt, const cytnx_uint16 &rc) {
      detail::check_tensor_initialized(Lt, "Mod");
      Storage Cnst(1, Type.Uint16);
      Cnst.at<cytnx_uint16>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Lt._impl->storage().size(),
                                     Type.type_promote(Type.Uint16, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(),Type.Uint16 < Lt.dtype()?Type.Uint16:Lt.dtype(),Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_uint16;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl,
                                               Lt._impl->storage()._impl, Cnst._impl,
                                               Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }
    template <>
    Tensor Mod<cytnx_bool>(const Tensor &Lt, const cytnx_bool &rc) {
      detail::check_tensor_initialized(Lt, "Mod");
      Storage Cnst(1, Type.Bool);
      Cnst.at<cytnx_bool>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Lt._impl->storage().size(), Type.type_promote(Type.Bool, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(),Type.Bool < Lt.dtype()?Type.Bool:Lt.dtype(),Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_bool;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl,
                                               Lt._impl->storage()._impl, Cnst._impl,
                                               Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Mod<Scalar>(const Tensor &Lt, const Scalar &rc) {
      detail::check_tensor_initialized(Lt, "Mod");
      Storage Cnst(1, rc.dtype());
      Cnst.set_item(0, rc);

      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Lt._impl->storage().size(), Type.type_promote(rc.dtype(), Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(),Type.Bool < Lt.dtype()?Type.Bool:Lt.dtype(),Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl, auto rhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::ModInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant(), Cnst.as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        cytnx::linalg_internal::cuMod_dispatch(out._impl->storage()._impl,
                                               Lt._impl->storage()._impl, Cnst._impl,
                                               Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Mod] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

  }  // namespace linalg

  Tensor operator%(const Tensor &Lt, const Tensor &Rt) { return cytnx::linalg::Mod(Lt, Rt); }
  template <>
  Tensor operator%<cytnx_complex128>(const cytnx_complex128 &lc, const Tensor &Rt) {
    return cytnx::linalg::Mod(lc, Rt);
  }
  template <>
  Tensor operator%<cytnx_complex64>(const cytnx_complex64 &lc, const Tensor &Rt) {
    return cytnx::linalg::Mod(lc, Rt);
  }
  template <>
  Tensor operator%<cytnx_double>(const cytnx_double &lc, const Tensor &Rt) {
    return cytnx::linalg::Mod(lc, Rt);
  }
  template <>
  Tensor operator%<cytnx_float>(const cytnx_float &lc, const Tensor &Rt) {
    return cytnx::linalg::Mod(lc, Rt);
  }
  template <>
  Tensor operator%<cytnx_int64>(const cytnx_int64 &lc, const Tensor &Rt) {
    return cytnx::linalg::Mod(lc, Rt);
  }
  template <>
  Tensor operator%<cytnx_uint64>(const cytnx_uint64 &lc, const Tensor &Rt) {
    return cytnx::linalg::Mod(lc, Rt);
  }
  template <>
  Tensor operator%<cytnx_int32>(const cytnx_int32 &lc, const Tensor &Rt) {
    return cytnx::linalg::Mod(lc, Rt);
  }
  template <>
  Tensor operator%<cytnx_uint32>(const cytnx_uint32 &lc, const Tensor &Rt) {
    return cytnx::linalg::Mod(lc, Rt);
  }
  template <>
  Tensor operator%<cytnx_int16>(const cytnx_int16 &lc, const Tensor &Rt) {
    return cytnx::linalg::Mod(lc, Rt);
  }
  template <>
  Tensor operator%<cytnx_uint16>(const cytnx_uint16 &lc, const Tensor &Rt) {
    return cytnx::linalg::Mod(lc, Rt);
  }
  template <>
  Tensor operator%<cytnx_bool>(const cytnx_bool &lc, const Tensor &Rt) {
    return cytnx::linalg::Mod(lc, Rt);
  }
  template <>
  Tensor operator%<Scalar>(const Scalar &lc, const Tensor &Rt) {
    return cytnx::linalg::Mod(lc, Rt);
  }

  template <>
  Tensor operator%<cytnx_complex128>(const Tensor &Lt, const cytnx_complex128 &rc) {
    return cytnx::linalg::Mod(Lt, rc);
  }
  template <>
  Tensor operator%<cytnx_complex64>(const Tensor &Lt, const cytnx_complex64 &rc) {
    return cytnx::linalg::Mod(Lt, rc);
  }
  template <>
  Tensor operator%<cytnx_double>(const Tensor &Lt, const cytnx_double &rc) {
    return cytnx::linalg::Mod(Lt, rc);
  }
  template <>
  Tensor operator%<cytnx_float>(const Tensor &Lt, const cytnx_float &rc) {
    return cytnx::linalg::Mod(Lt, rc);
  }
  template <>
  Tensor operator%<cytnx_int64>(const Tensor &Lt, const cytnx_int64 &rc) {
    return cytnx::linalg::Mod(Lt, rc);
  }
  template <>
  Tensor operator%<cytnx_uint64>(const Tensor &Lt, const cytnx_uint64 &rc) {
    return cytnx::linalg::Mod(Lt, rc);
  }
  template <>
  Tensor operator%<cytnx_int32>(const Tensor &Lt, const cytnx_int32 &rc) {
    return cytnx::linalg::Mod(Lt, rc);
  }
  template <>
  Tensor operator%<cytnx_uint32>(const Tensor &Lt, const cytnx_uint32 &rc) {
    return cytnx::linalg::Mod(Lt, rc);
  }
  template <>
  Tensor operator%<cytnx_int16>(const Tensor &Lt, const cytnx_int16 &rc) {
    return cytnx::linalg::Mod(Lt, rc);
  }
  template <>
  Tensor operator%<cytnx_uint16>(const Tensor &Lt, const cytnx_uint16 &rc) {
    return cytnx::linalg::Mod(Lt, rc);
  }
  template <>
  Tensor operator%<cytnx_bool>(const Tensor &Lt, const cytnx_bool &rc) {
    return cytnx::linalg::Mod(Lt, rc);
  }
  template <>
  Tensor operator%<Scalar>(const Tensor &Lt, const Scalar &rc) {
    return cytnx::linalg::Mod(Lt, rc);
  }

}  // namespace cytnx

namespace cytnx {
  namespace linalg {
    //===============
    // cytnx::UniTensor
    //===============
    cytnx::UniTensor Mod(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt) {
      cytnx_error_msg(true, "[Mod][Developing]%s", "\n");
      return cytnx::UniTensor();
    }

    template <class T>
    cytnx::UniTensor Mod(const T &lc, const cytnx::UniTensor &Rt) {
      cytnx::UniTensor out = Rt.clone();
      if (out.uten_type() == UTenType.Dense) {
        out.get_block_() = cytnx::linalg::Mod(lc, out.get_block_());
      } else {
        cytnx_error_msg(out.uten_type() == UTenType.Void,
                        "[ERROR] UniTensor is not initialized and of type Void.%s", "\n");
        cytnx_error_msg(
          out.uten_type() == UTenType.Sparse,
          "[ERROR] SparseUniTensor is deprecated. Use BlockUniTensor or LinOp instead.%s", "\n");
        cytnx_error_msg(true, "[ERROR] UniTensor type '%s' is currently not supported\n",
                        out.uten_type_str().c_str());
      }
      return out;
    }

    template cytnx::UniTensor Mod<cytnx_complex128>(const cytnx_complex128 &lc,
                                                    const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Mod<cytnx_complex64>(const cytnx_complex64 &lc,
                                                   const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Mod<cytnx_double>(const cytnx_double &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Mod<cytnx_float>(const cytnx_float &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Mod<cytnx_int64>(const cytnx_int64 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Mod<cytnx_uint64>(const cytnx_uint64 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Mod<cytnx_int32>(const cytnx_int32 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Mod<cytnx_uint32>(const cytnx_uint32 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Mod<cytnx_int16>(const cytnx_int16 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Mod<cytnx_uint16>(const cytnx_uint16 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Mod<cytnx_bool>(const cytnx_bool &lc, const cytnx::UniTensor &Rt);

    template <class T>
    cytnx::UniTensor Mod(const cytnx::UniTensor &Lt, const T &rc) {
      cytnx::UniTensor out = Lt.clone();
      if (out.uten_type() == UTenType.Dense) {
        out.get_block_() = cytnx::linalg::Mod(out.get_block_(), rc);
      } else {
        cytnx_error_msg(out.uten_type() == UTenType.Void,
                        "[ERROR] UniTensor is not initialized and of type Void.%s", "\n");
        cytnx_error_msg(
          out.uten_type() == UTenType.Sparse,
          "[ERROR] SparseUniTensor is deprecated. Use BlockUniTensor or LinOp instead.%s", "\n");
        cytnx_error_msg(true, "[ERROR] UniTensor type '%s' is currently not supported\n",
                        out.uten_type_str().c_str());
      }
      return out;
    }

    template cytnx::UniTensor Mod<cytnx_complex128>(const cytnx::UniTensor &Lt,
                                                    const cytnx_complex128 &rc);
    template cytnx::UniTensor Mod<cytnx_complex64>(const cytnx::UniTensor &Lt,
                                                   const cytnx_complex64 &rc);
    template cytnx::UniTensor Mod<cytnx_double>(const cytnx::UniTensor &Lt, const cytnx_double &rc);
    template cytnx::UniTensor Mod<cytnx_float>(const cytnx::UniTensor &Lt, const cytnx_float &rc);
    template cytnx::UniTensor Mod<cytnx_int64>(const cytnx::UniTensor &Lt, const cytnx_int64 &rc);
    template cytnx::UniTensor Mod<cytnx_uint64>(const cytnx::UniTensor &Lt, const cytnx_uint64 &rc);
    template cytnx::UniTensor Mod<cytnx_int32>(const cytnx::UniTensor &Lt, const cytnx_int32 &rc);
    template cytnx::UniTensor Mod<cytnx_uint32>(const cytnx::UniTensor &Lt, const cytnx_uint32 &rc);
    template cytnx::UniTensor Mod<cytnx_int16>(const cytnx::UniTensor &Lt, const cytnx_int16 &rc);
    template cytnx::UniTensor Mod<cytnx_uint16>(const cytnx::UniTensor &Lt, const cytnx_uint16 &rc);
    template cytnx::UniTensor Mod<cytnx_bool>(const cytnx::UniTensor &Lt, const cytnx_bool &rc);
  }  // namespace linalg

  //=================
  // cytnx::UniTensor
  //=================
  cytnx::UniTensor operator%(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt) {
    return cytnx::linalg::Mod(Lt, Rt);
  }

  template <cytnx_scalar_like T>
  cytnx::UniTensor operator%(const T &lc, const cytnx::UniTensor &Rt) {
    return cytnx::linalg::Mod(lc, Rt);
  }

  template cytnx::UniTensor operator%
    <cytnx_complex128>(const cytnx_complex128 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator%
    <cytnx_complex64>(const cytnx_complex64 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator%
    <cytnx_double>(const cytnx_double &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator%
    <cytnx_float>(const cytnx_float &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator%
    <cytnx_int64>(const cytnx_int64 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator%
    <cytnx_uint64>(const cytnx_uint64 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator%
    <cytnx_int32>(const cytnx_int32 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator%
    <cytnx_uint32>(const cytnx_uint32 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator%
    <cytnx_int16>(const cytnx_int16 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator%
    <cytnx_uint16>(const cytnx_uint16 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator%<cytnx_bool>(const cytnx_bool &lc, const cytnx::UniTensor &Rt);

  template <cytnx_scalar_like T>
  cytnx::UniTensor operator%(const cytnx::UniTensor &Lt, const T &rc) {
    return cytnx::linalg::Mod(Lt, rc);
  }

  template cytnx::UniTensor operator%
    <cytnx_complex128>(const cytnx::UniTensor &Lt, const cytnx_complex128 &rc);
  template cytnx::UniTensor operator%
    <cytnx_complex64>(const cytnx::UniTensor &Lt, const cytnx_complex64 &rc);
  template cytnx::UniTensor operator%
    <cytnx_double>(const cytnx::UniTensor &Lt, const cytnx_double &rc);
  template cytnx::UniTensor operator%
    <cytnx_float>(const cytnx::UniTensor &Lt, const cytnx_float &rc);
  template cytnx::UniTensor operator%
    <cytnx_int64>(const cytnx::UniTensor &Lt, const cytnx_int64 &rc);
  template cytnx::UniTensor operator%
    <cytnx_uint64>(const cytnx::UniTensor &Lt, const cytnx_uint64 &rc);
  template cytnx::UniTensor operator%
    <cytnx_int32>(const cytnx::UniTensor &Lt, const cytnx_int32 &rc);
  template cytnx::UniTensor operator%
    <cytnx_uint32>(const cytnx::UniTensor &Lt, const cytnx_uint32 &rc);
  template cytnx::UniTensor operator%
    <cytnx_int16>(const cytnx::UniTensor &Lt, const cytnx_int16 &rc);
  template cytnx::UniTensor operator%
    <cytnx_uint16>(const cytnx::UniTensor &Lt, const cytnx_uint16 &rc);
  template cytnx::UniTensor operator%<cytnx_bool>(const cytnx::UniTensor &Lt, const cytnx_bool &rc);

}  // namespace cytnx

#endif
