#include "linalg.hpp"

#include "Tensor.hpp"
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "Arithmetic_shape.hpp"
  #include "backend/linalg_internal_interface.hpp"
namespace cytnx {
  namespace linalg {
    Tensor Sub(const Tensor &Lt, const Tensor &Rt) {
      detail::check_binary_tensor_inputs(Lt, Rt, "Sub");
      cytnx_error_msg(Lt.device() != Rt.device(),
                      "[Sub] The two tensors cannot be on different devices.%s", "\n");

      Tensor out;
      bool icnst = false;
      if (detail::init_broadcast_binary_output(out, Lt, Rt,
                                               Type.type_promote(Lt.dtype(), Rt.dtype()))) {
        icnst = true;
      } else {
        cytnx_error_msg(Lt.shape() != Rt.shape(),
                        "[Sub] The two tensors do not have the same shape.%s", "\n");
        out.Init(Lt.shape(), Type.type_promote(Lt.dtype(), Rt.dtype()), Lt.device());
      }

      if (out.storage().size() == 0) return out;

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
              cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
                storage_cast<TO>(out._impl->storage()._impl), lhs_impl, rhs_impl,
                out._impl->storage().size(), {}, {}, {});
            },
            Lt._impl->storage().as_storage_variant(), Rt._impl->storage().as_storage_variant());
        } else {
  #ifdef UNI_GPU
          checkCudaErrors(cudaSetDevice(Rt.device()));
          linalg_internal::cuSub_dispatch(out._impl->storage()._impl, left._impl->storage()._impl,
                                          right._impl->storage()._impl,
                                          out._impl->storage()._impl->size(), {}, {}, {});
  #else
          cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
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
              cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
                storage_cast<TO>(out._impl->storage()._impl), lhs_impl, rhs_impl,
                out._impl->storage().size(), Lt._impl->shape(), Lt._impl->invmapper(),
                Rt._impl->invmapper());
            },
            Lt._impl->storage().as_storage_variant(), Rt._impl->storage().as_storage_variant());
        } else {
  #ifdef UNI_GPU
          checkCudaErrors(cudaSetDevice(Rt.device()));
          linalg_internal::cuSub_dispatch(out._impl->storage()._impl, left._impl->storage()._impl,
                                          right._impl->storage()._impl,
                                          out._impl->storage()._impl->size(), left._impl->shape(),
                                          left._impl->invmapper(), right._impl->invmapper());
  #else
          cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
  #endif
        }
      }
      return out;
    }

    //-----------------------------------------------------------------------------------
    template <>
    Tensor Sub<cytnx_complex128>(const cytnx_complex128 &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Sub");
      Storage Cnst(1, Type.ComplexDouble);
      Cnst.at<cytnx_complex128>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Rt._impl->storage().size(), Type.ComplexDouble, Rt.device());
      // Tensor out(Rt.shape(), Type.ComplexDouble, Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_complex128;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Cnst._impl,
                                        Rt._impl->storage()._impl,
                                        Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_complex64>(const cytnx_complex64 &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Sub");
      Storage Cnst(1, Type.ComplexFloat);
      Cnst.at<cytnx_complex64>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Rt._impl->storage().size(),
                                     Type.type_promote(Type.ComplexFloat, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(), Type.type_promote(Type.ComplexFloat, Rt.dtype()),
      // Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_complex64;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Cnst._impl,
                                        Rt._impl->storage()._impl,
                                        Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_double>(const cytnx_double &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Sub");
      Storage Cnst(1, Type.Double);
      Cnst.at<cytnx_double>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Rt._impl->storage().size(),
                                     Type.type_promote(Type.Double, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(), Type.type_promote(Type.Double, Rt.dtype()), Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_double;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Cnst._impl,
                                        Rt._impl->storage()._impl,
                                        Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_float>(const cytnx_float &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Sub");
      Storage Cnst(1, Type.Float);
      Cnst.at<cytnx_float>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.type_promote(Type.Float, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(), Type.type_promote(Type.Float, Rt.dtype()), Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_float;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Cnst._impl,
                                        Rt._impl->storage()._impl,
                                        Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_int64>(const cytnx_int64 &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Sub");
      Storage Cnst(1, Type.Int64);
      Cnst.at<cytnx_int64>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.type_promote(Type.Int64, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(), Type.type_promote(Type.Int64, Rt.dtype()), Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_int64;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Cnst._impl,
                                        Rt._impl->storage()._impl,
                                        Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_uint64>(const cytnx_uint64 &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Sub");
      Storage Cnst(1, Type.Uint64);
      Cnst.at<cytnx_uint64>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Rt._impl->storage().size(),
                                     Type.type_promote(Type.Uint64, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(), Type.type_promote(Type.Uint64, Rt.dtype()), Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_uint64;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Cnst._impl,
                                        Rt._impl->storage()._impl,
                                        Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_int32>(const cytnx_int32 &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Sub");
      Storage Cnst(1, Type.Int32);
      Cnst.at<cytnx_int32>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.type_promote(Type.Int32, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(), Type.type_promote(Type.Int32, Rt.dtype()), Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_int32;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Cnst._impl,
                                        Rt._impl->storage()._impl,
                                        Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_uint32>(const cytnx_uint32 &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Sub");
      Storage Cnst(1, Type.Uint32);
      Cnst.at<cytnx_uint32>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Rt._impl->storage().size(),
                                     Type.type_promote(Type.Uint32, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(), Type.type_promote(Type.Uint32, Rt.dtype()), Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_uint32;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Cnst._impl,
                                        Rt._impl->storage()._impl,
                                        Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_int16>(const cytnx_int16 &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Sub");
      Storage Cnst(1, Type.Int16);
      Cnst.at<cytnx_int16>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.type_promote(Type.Int16, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(), Type.type_promote(Type.Int16, Rt.dtype()), Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_int16;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Cnst._impl,
                                        Rt._impl->storage()._impl,
                                        Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_uint16>(const cytnx_uint16 &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Sub");
      Storage Cnst(1, Type.Uint16);
      Cnst.at<cytnx_uint16>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Rt._impl->storage().size(),
                                     Type.type_promote(Type.Uint16, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(), Type.type_promote(Type.Uint16, Rt.dtype()), Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_uint16;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Cnst._impl,
                                        Rt._impl->storage()._impl,
                                        Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_bool>(const cytnx_bool &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Sub");
      Storage Cnst(1, Type.Bool);
      Cnst.at<cytnx_bool>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.type_promote(Type.Bool, Rt.dtype()), Rt.device());
      // Tensor out(Rt.shape(), Type.type_promote(Type.Bool, Rt.dtype()), Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto rhs_impl) {
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TL = cytnx_bool;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), storage_cast<TL>(Cnst._impl), rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Cnst._impl,
                                        Rt._impl->storage()._impl,
                                        Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<Scalar>(const Scalar &lc, const Tensor &Rt) {
      detail::check_tensor_initialized(Rt, "Sub");
      Storage Cnst(1, lc.dtype());
      Cnst.set_item(0, lc);

      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.type_promote(lc.dtype(), Rt.dtype()), Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl, auto rhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Cnst.as_storage_variant(), Rt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Cnst._impl,
                                        Rt._impl->storage()._impl,
                                        Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    //-----------------------------------------------------------------------------------
    template <>
    Tensor Sub<cytnx_complex128>(const Tensor &Lt, const cytnx_complex128 &rc) {
      detail::check_tensor_initialized(Lt, "Sub");
      Storage Cnst(1, Type.ComplexDouble);
      Cnst.at<cytnx_complex128>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Lt._impl->storage().size(), Type.ComplexDouble, Lt.device());
      // Tensor out(Lt.shape(), Type.ComplexDouble, Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_complex128;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Lt._impl->storage()._impl,
                                        Cnst._impl, Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_complex64>(const Tensor &Lt, const cytnx_complex64 &rc) {
      detail::check_tensor_initialized(Lt, "Sub");
      Storage Cnst(1, Type.ComplexFloat);
      Cnst.at<cytnx_complex64>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Lt._impl->storage().size(),
                                     Type.type_promote(Type.ComplexFloat, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(), Type.type_promote(Type.ComplexFloat, Lt.dtype()),
      // Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_complex64;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Lt._impl->storage()._impl,
                                        Cnst._impl, Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_double>(const Tensor &Lt, const cytnx_double &rc) {
      detail::check_tensor_initialized(Lt, "Sub");
      Storage Cnst(1, Type.Double);
      Cnst.at<cytnx_double>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Lt._impl->storage().size(),
                                     Type.type_promote(Type.Double, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(), Type.type_promote(Type.Double, Lt.dtype()), Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_double;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Lt._impl->storage()._impl,
                                        Cnst._impl, Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_float>(const Tensor &Lt, const cytnx_float &rc) {
      detail::check_tensor_initialized(Lt, "Sub");
      Storage Cnst(1, Type.Float);
      Cnst.at<cytnx_float>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Lt._impl->storage().size(), Type.type_promote(Type.Float, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(), Type.type_promote(Type.Float, Lt.dtype()), Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_float;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Lt._impl->storage()._impl,
                                        Cnst._impl, Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_int64>(const Tensor &Lt, const cytnx_int64 &rc) {
      detail::check_tensor_initialized(Lt, "Sub");
      Storage Cnst(1, Type.Int64);
      Cnst.at<cytnx_int64>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Lt._impl->storage().size(), Type.type_promote(Type.Int64, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(), Type.type_promote(Type.Int64, Lt.dtype()), Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_int64;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Lt._impl->storage()._impl,
                                        Cnst._impl, Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_uint64>(const Tensor &Lt, const cytnx_uint64 &rc) {
      detail::check_tensor_initialized(Lt, "Sub");
      Storage Cnst(1, Type.Uint64);
      Cnst.at<cytnx_uint64>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Lt._impl->storage().size(),
                                     Type.type_promote(Type.Uint64, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(), Type.type_promote(Type.Uint64, Lt.dtype()), Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_uint64;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Lt._impl->storage()._impl,
                                        Cnst._impl, Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_int32>(const Tensor &Lt, const cytnx_int32 &rc) {
      detail::check_tensor_initialized(Lt, "Sub");
      Storage Cnst(1, Type.Int32);
      Cnst.at<cytnx_int32>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Lt._impl->storage().size(), Type.type_promote(Type.Int32, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(), Type.type_promote(Type.Int32, Lt.dtype()), Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_int32;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Lt._impl->storage()._impl,
                                        Cnst._impl, Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_uint32>(const Tensor &Lt, const cytnx_uint32 &rc) {
      detail::check_tensor_initialized(Lt, "Sub");
      Storage Cnst(1, Type.Uint32);
      Cnst.at<cytnx_uint32>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Lt._impl->storage().size(),
                                     Type.type_promote(Type.Uint32, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(), Type.type_promote(Type.Uint32, Lt.dtype()), Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_uint32;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Lt._impl->storage()._impl,
                                        Cnst._impl, Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_int16>(const Tensor &Lt, const cytnx_int16 &rc) {
      detail::check_tensor_initialized(Lt, "Sub");
      Storage Cnst(1, Type.Int16);
      Cnst.at<cytnx_int16>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Lt._impl->storage().size(), Type.type_promote(Type.Int16, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(), Type.type_promote(Type.Int16, Lt.dtype()), Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_int16;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Lt._impl->storage()._impl,
                                        Cnst._impl, Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_uint16>(const Tensor &Lt, const cytnx_uint16 &rc) {
      detail::check_tensor_initialized(Lt, "Sub");
      Storage Cnst(1, Type.Uint16);
      Cnst.at<cytnx_uint16>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Lt._impl->storage().size(),
                                     Type.type_promote(Type.Uint16, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(), Type.type_promote(Type.Uint16, Lt.dtype()), Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_uint16;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Lt._impl->storage()._impl,
                                        Cnst._impl, Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<cytnx_bool>(const Tensor &Lt, const cytnx_bool &rc) {
      detail::check_tensor_initialized(Lt, "Sub");
      Storage Cnst(1, Type.Bool);
      Cnst.at<cytnx_bool>(0) = rc;
      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Lt._impl->storage().size(), Type.type_promote(Type.Bool, Lt.dtype()), Lt.device());
      // Tensor out(Lt.shape(), Type.type_promote(Type.Bool, Lt.dtype()), Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = cytnx_bool;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, storage_cast<TR>(Cnst._impl),
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Lt._impl->storage()._impl,
                                        Cnst._impl, Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    template <>
    Tensor Sub<Scalar>(const Tensor &Lt, const Scalar &rc) {
      detail::check_tensor_initialized(Lt, "Sub");
      Storage Cnst(1, rc.dtype());
      Cnst.set_item(0, rc);

      Tensor out;
      out._impl = Lt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Lt._impl->storage().size(), Type.type_promote(Lt.dtype(), rc.dtype()), Lt.device());

      if (Lt.device() == Device.cpu) {
        std::visit(
          [&](auto lhs_impl, auto rhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = storage_value_t<decltype(rhs_impl)>;
            using TO = Type_class::type_promote_t<TL, TR>;
            cytnx::linalg_internal::SubInternalImpl<TO, TL, TR>(
              storage_cast<TO>(out._impl->storage()._impl), lhs_impl, rhs_impl,
              out._impl->storage().size(), {}, {}, {});
          },
          Lt._impl->storage().as_storage_variant(), Cnst.as_storage_variant());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Lt.device()));
        linalg_internal::cuSub_dispatch(out._impl->storage()._impl, Lt._impl->storage()._impl,
                                        Cnst._impl, Lt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }
      return out;
    }

  }  // namespace linalg

  //==================
  // operator
  //==================

  Tensor operator-(const Tensor &Lt, const Tensor &Rt) { return cytnx::linalg::Sub(Lt, Rt); }

  template <>
  Tensor operator-<cytnx_complex128>(const cytnx_complex128 &lc, const Tensor &Rt) {
    return cytnx::linalg::Sub(lc, Rt);
  }
  template <>
  Tensor operator-<cytnx_complex64>(const cytnx_complex64 &lc, const Tensor &Rt) {
    return cytnx::linalg::Sub(lc, Rt);
  }
  template <>
  Tensor operator-<cytnx_double>(const cytnx_double &lc, const Tensor &Rt) {
    return cytnx::linalg::Sub(lc, Rt);
  }
  template <>
  Tensor operator-<cytnx_float>(const cytnx_float &lc, const Tensor &Rt) {
    return cytnx::linalg::Sub(lc, Rt);
  }
  template <>
  Tensor operator-<cytnx_int64>(const cytnx_int64 &lc, const Tensor &Rt) {
    return cytnx::linalg::Sub(lc, Rt);
  }
  template <>
  Tensor operator-<cytnx_uint64>(const cytnx_uint64 &lc, const Tensor &Rt) {
    return cytnx::linalg::Sub(lc, Rt);
  }
  template <>
  Tensor operator-<cytnx_int32>(const cytnx_int32 &lc, const Tensor &Rt) {
    return cytnx::linalg::Sub(lc, Rt);
  }
  template <>
  Tensor operator-<cytnx_uint32>(const cytnx_uint32 &lc, const Tensor &Rt) {
    return cytnx::linalg::Sub(lc, Rt);
  }
  template <>
  Tensor operator-<cytnx_int16>(const cytnx_int16 &lc, const Tensor &Rt) {
    return cytnx::linalg::Sub(lc, Rt);
  }
  template <>
  Tensor operator-<cytnx_uint16>(const cytnx_uint16 &lc, const Tensor &Rt) {
    return cytnx::linalg::Sub(lc, Rt);
  }
  template <>
  Tensor operator-<cytnx_bool>(const cytnx_bool &lc, const Tensor &Rt) {
    return cytnx::linalg::Sub(lc, Rt);
  }
  template <>
  Tensor operator-<Tensor::Tproxy>(const Tensor::Tproxy &lc, const Tensor &Rt) {
    return Tensor(lc) - Rt;
  }
  template <>
  Tensor operator-<Scalar>(const Scalar &lc, const Tensor &Rt) {
    return cytnx::linalg::Sub(lc, Rt);
  }
  template <>
  Tensor operator-<Scalar::Sproxy>(const Scalar::Sproxy &lc, const Tensor &Rt) {
    return cytnx::linalg::Sub(Scalar(lc), Rt);
  }

  template <>
  Tensor operator-<cytnx_complex128>(const Tensor &Lt, const cytnx_complex128 &rc) {
    return cytnx::linalg::Sub(Lt, rc);
  }
  template <>
  Tensor operator-<cytnx_complex64>(const Tensor &Lt, const cytnx_complex64 &rc) {
    return cytnx::linalg::Sub(Lt, rc);
  }
  template <>
  Tensor operator-<cytnx_double>(const Tensor &Lt, const cytnx_double &rc) {
    return cytnx::linalg::Sub(Lt, rc);
  }
  template <>
  Tensor operator-<cytnx_float>(const Tensor &Lt, const cytnx_float &rc) {
    return cytnx::linalg::Sub(Lt, rc);
  }
  template <>
  Tensor operator-<cytnx_int64>(const Tensor &Lt, const cytnx_int64 &rc) {
    return cytnx::linalg::Sub(Lt, rc);
  }
  template <>
  Tensor operator-<cytnx_uint64>(const Tensor &Lt, const cytnx_uint64 &rc) {
    return cytnx::linalg::Sub(Lt, rc);
  }
  template <>
  Tensor operator-<cytnx_int32>(const Tensor &Lt, const cytnx_int32 &rc) {
    return cytnx::linalg::Sub(Lt, rc);
  }
  template <>
  Tensor operator-<cytnx_uint32>(const Tensor &Lt, const cytnx_uint32 &rc) {
    return cytnx::linalg::Sub(Lt, rc);
  }
  template <>
  Tensor operator-<cytnx_int16>(const Tensor &Lt, const cytnx_int16 &rc) {
    return cytnx::linalg::Sub(Lt, rc);
  }
  template <>
  Tensor operator-<cytnx_uint16>(const Tensor &Lt, const cytnx_uint16 &rc) {
    return cytnx::linalg::Sub(Lt, rc);
  }
  template <>
  Tensor operator-<cytnx_bool>(const Tensor &Lt, const cytnx_bool &rc) {
    return cytnx::linalg::Sub(Lt, rc);
  }
  template <>
  Tensor operator-<Scalar>(const Tensor &Lt, const Scalar &rc) {
    return cytnx::linalg::Sub(Lt, rc);
  }

}  // namespace cytnx

namespace cytnx {
  namespace linalg {
    //===============
    // cytnx: cytnx::UniTensor
    //===============
    cytnx::UniTensor Sub(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt) {
      // promote across the real/complex boundary (e.g. ComplexFloat - Double -> ComplexDouble)
      // rather than adopting the lower-enum operand dtype.
      UniTensor out = Lt.clone();
      const unsigned int out_dtype = Type.type_promote(Lt.dtype(), Rt.dtype());
      if (out.dtype() != out_dtype) out = out.astype(out_dtype);
      out.relabel_(vec_range<std::string>(Lt.rank()));
      out.set_name_("");

      out.Sub_(Rt);

      return out;
    }

    template <class T>
    cytnx::UniTensor Sub(const T &lc, const cytnx::UniTensor &Rt) {
      // cytnx_error_msg(Rt.is_tag(),"[ERROR] Cannot perform arithmetic on tagged
      // unitensor.%s","\n");

      UniTensor out = Rt.clone();
      const unsigned int out_dtype = Type.type_promote(Scalar(lc).dtype(), Rt.dtype());
      if (out.dtype() != out_dtype) out = out.astype(out_dtype);
      out._impl->lSub_(lc);
      out.set_name_("");

      return out;
    }

    template cytnx::UniTensor Sub<cytnx_complex128>(const cytnx_complex128 &lc,
                                                    const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Sub<cytnx_complex64>(const cytnx_complex64 &lc,
                                                   const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Sub<cytnx_double>(const cytnx_double &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Sub<cytnx_float>(const cytnx_float &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Sub<cytnx_int64>(const cytnx_int64 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Sub<cytnx_uint64>(const cytnx_uint64 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Sub<cytnx_int32>(const cytnx_int32 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Sub<cytnx_uint32>(const cytnx_uint32 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Sub<cytnx_int16>(const cytnx_int16 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Sub<cytnx_uint16>(const cytnx_uint16 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Sub<cytnx_bool>(const cytnx_bool &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Sub<Scalar>(const Scalar &lc, const cytnx::UniTensor &Rt);

    template <class T>
    cytnx::UniTensor Sub(const cytnx::UniTensor &Lt, const T &rc) {
      // cytnx_error_msg(Lt.is_tag(),"[ERROR] Cannot perform arithmetic on tagged
      // unitensor.%s","\n");

      UniTensor out = Lt.clone();
      const unsigned int out_dtype = Type.type_promote(Lt.dtype(), Scalar(rc).dtype());
      if (out.dtype() != out_dtype) out = out.astype(out_dtype);
      out.Sub_(rc);
      // out.relabel_(vec_range<cytnx_int64>(Lt.rank()));
      out.set_name_("");

      return out;
    }

    template cytnx::UniTensor Sub<cytnx_complex128>(const cytnx::UniTensor &Lt,
                                                    const cytnx_complex128 &rc);
    template cytnx::UniTensor Sub<cytnx_complex64>(const cytnx::UniTensor &Lt,
                                                   const cytnx_complex64 &rc);
    template cytnx::UniTensor Sub<cytnx_double>(const cytnx::UniTensor &Lt, const cytnx_double &rc);
    template cytnx::UniTensor Sub<cytnx_float>(const cytnx::UniTensor &Lt, const cytnx_float &rc);
    template cytnx::UniTensor Sub<cytnx_int64>(const cytnx::UniTensor &Lt, const cytnx_int64 &rc);
    template cytnx::UniTensor Sub<cytnx_uint64>(const cytnx::UniTensor &Lt, const cytnx_uint64 &rc);
    template cytnx::UniTensor Sub<cytnx_int32>(const cytnx::UniTensor &Lt, const cytnx_int32 &rc);
    template cytnx::UniTensor Sub<cytnx_uint32>(const cytnx::UniTensor &Lt, const cytnx_uint32 &rc);
    template cytnx::UniTensor Sub<cytnx_int16>(const cytnx::UniTensor &Lt, const cytnx_int16 &rc);
    template cytnx::UniTensor Sub<cytnx_uint16>(const cytnx::UniTensor &Lt, const cytnx_uint16 &rc);
    template cytnx::UniTensor Sub<cytnx_bool>(const cytnx::UniTensor &Lt, const cytnx_bool &rc);
    template cytnx::UniTensor Sub<Scalar>(const cytnx::UniTensor &Lt, const Scalar &rc);

  }  // namespace linalg

  //=================
  // Cytnx: cytnx::UniTensor
  //=================
  cytnx::UniTensor operator-(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt) {
    return cytnx::linalg::Sub(Lt, Rt);
  }

  template <class T>
  cytnx::UniTensor operator-(const T &lc, const cytnx::UniTensor &Rt) {
    return cytnx::linalg::Sub(lc, Rt);
  }
  template <>
  cytnx::UniTensor operator-<Scalar::Sproxy>(const Scalar::Sproxy &lc, const cytnx::UniTensor &Rt) {
    return cytnx::linalg::Sub(Scalar(lc), Rt);
  }

  template cytnx::UniTensor operator-
    <cytnx_complex128>(const cytnx_complex128 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator-
    <cytnx_complex64>(const cytnx_complex64 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator-
    <cytnx_double>(const cytnx_double &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator-
    <cytnx_float>(const cytnx_float &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator-
    <cytnx_int64>(const cytnx_int64 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator-
    <cytnx_uint64>(const cytnx_uint64 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator-
    <cytnx_int32>(const cytnx_int32 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator-
    <cytnx_uint32>(const cytnx_uint32 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator-
    <cytnx_int16>(const cytnx_int16 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator-
    <cytnx_uint16>(const cytnx_uint16 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator-<cytnx_bool>(const cytnx_bool &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator-<Scalar>(const Scalar &lc, const cytnx::UniTensor &Rt);

  template <class T>
  cytnx::UniTensor operator-(const cytnx::UniTensor &Lt, const T &rc) {
    return cytnx::linalg::Sub(Lt, rc);
  }
  template <>
  cytnx::UniTensor operator-<Scalar::Sproxy>(const cytnx::UniTensor &Lt, const Scalar::Sproxy &rc) {
    return cytnx::linalg::Sub(Lt, Scalar(rc));
  }
  template cytnx::UniTensor operator-
    <cytnx_complex128>(const cytnx::UniTensor &Lt, const cytnx_complex128 &rc);
  template cytnx::UniTensor operator-
    <cytnx_complex64>(const cytnx::UniTensor &Lt, const cytnx_complex64 &rc);
  template cytnx::UniTensor operator-
    <cytnx_double>(const cytnx::UniTensor &Lt, const cytnx_double &rc);
  template cytnx::UniTensor operator-
    <cytnx_float>(const cytnx::UniTensor &Lt, const cytnx_float &rc);
  template cytnx::UniTensor operator-
    <cytnx_int64>(const cytnx::UniTensor &Lt, const cytnx_int64 &rc);
  template cytnx::UniTensor operator-
    <cytnx_uint64>(const cytnx::UniTensor &Lt, const cytnx_uint64 &rc);
  template cytnx::UniTensor operator-
    <cytnx_int32>(const cytnx::UniTensor &Lt, const cytnx_int32 &rc);
  template cytnx::UniTensor operator-
    <cytnx_uint32>(const cytnx::UniTensor &Lt, const cytnx_uint32 &rc);
  template cytnx::UniTensor operator-
    <cytnx_int16>(const cytnx::UniTensor &Lt, const cytnx_int16 &rc);
  template cytnx::UniTensor operator-
    <cytnx_uint16>(const cytnx::UniTensor &Lt, const cytnx_uint16 &rc);
  template cytnx::UniTensor operator-<cytnx_bool>(const cytnx::UniTensor &Lt, const cytnx_bool &rc);
  template cytnx::UniTensor operator-<Scalar>(const cytnx::UniTensor &Lt, const Scalar &rc);

}  // namespace cytnx

#endif  // BACKEND_TORCH
