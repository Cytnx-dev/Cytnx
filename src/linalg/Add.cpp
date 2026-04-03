#include "linalg.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else

  #include "backend/linalg_internal_interface.hpp"
namespace cytnx {
  namespace linalg {
    namespace detail {
      inline unsigned int SelectAddOutputType(const unsigned int lhs_dtype,
                                              const unsigned int rhs_dtype, const int device) {
        (void)device;
        return Type.type_promote(lhs_dtype, rhs_dtype);
      }

      template <typename TL>
      Tensor AddScalarTensorImpl(const TL &lc, const Tensor &Rt) {
        const unsigned int lhs_dtype = Type.cy_typeid_v<TL>;
        Storage Cnst(1, lhs_dtype);
        Cnst.at<TL>(0) = lc;

        Tensor out;
        out._impl = Rt._impl->_clone_meta_only();
        out._impl->storage() =
          Storage(Rt._impl->storage().size(),
                  SelectAddOutputType(lhs_dtype, Rt.dtype(), Rt.device()), Rt.device());

        if (Rt.device() == Device.cpu) {
          std::visit(
            [&](auto *rptr) {
              using TR = std::remove_pointer_t<decltype(rptr)>;
              cytnx::linalg_internal::AddInternalImpl<TL, TR>(
                out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
                Rt._impl->storage()._impl->size(), {}, {}, {});
            },
            Rt.ptr());
        } else {
  #ifdef UNI_GPU
          checkCudaErrors(cudaSetDevice(Rt.device()));
          linalg_internal::cuAdd_dispatch(out._impl->storage()._impl, Cnst._impl,
                                          Rt._impl->storage()._impl,
                                          Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
          cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
  #endif
        }

        return out;
      }
    }  // namespace detail

    Tensor Add(const Tensor &Lt, const Tensor &Rt) {
      cytnx_error_msg(Lt.device() != Rt.device(),
                      "[Add] The two tensors cannot be on different devices.%s", "\n");

      const unsigned int out_dtype =
        detail::SelectAddOutputType(Lt.dtype(), Rt.dtype(), Lt.device());

      Tensor out;
      bool icnst = false;

      if (Lt.shape().size() == 1 && Lt.shape()[0] == 1) {
        out._impl = Rt._impl->_clone_meta_only();
        out._impl->storage() = Storage(Rt.storage().size(), out_dtype, Rt.device());
        icnst = true;

      } else if (Rt.shape().size() == 1 && Rt.shape()[0] == 1) {
        out._impl = Lt._impl->_clone_meta_only();
        out._impl->storage() = Storage(Lt.storage().size(), out_dtype, Lt.device());
        icnst = true;

      } else {
        cytnx_error_msg(Lt.shape() != Rt.shape(),
                        "[Add] The two tensors do not have the same shape. Lt rank: [%d] "
                        "Rt rank: [%d] %s",
                        Lt.shape().size(), Rt.shape().size(), "\n");
        out.Init(Lt.shape(), out_dtype, Lt.device());
      }

      // if contiguous, then no need to calculate the mappers
      if ((Lt.is_contiguous() && Rt.is_contiguous()) || icnst) {
        // contiguous section.
        if (Lt.device() == Device.cpu) {
          std::visit(
            [&](auto *lptr) {
              using TL = std::remove_pointer_t<decltype(lptr)>;
              std::visit(
                [&](auto *rptr) {
                  using TR = std::remove_pointer_t<decltype(rptr)>;
                  cytnx::linalg_internal::AddInternalImpl<TL, TR>(
                    out._impl->storage()._impl, Lt._impl->storage()._impl,
                    Rt._impl->storage()._impl, out._impl->storage()._impl->size(), {}, {}, {});
                },
                Rt.ptr());
            },
            Lt.ptr());
        } else {
  #ifdef UNI_GPU
          checkCudaErrors(cudaSetDevice(Rt.device()));
          linalg_internal::cuAdd_dispatch(out._impl->storage()._impl, Lt._impl->storage()._impl,
                                          Rt._impl->storage()._impl,
                                          out._impl->storage()._impl->size(), {}, {}, {});
  #else
          cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
  #endif
        }
      } else {
        // non-contiguous section
        if (Lt.device() == Device.cpu) {
          std::visit(
            [&](auto *lptr) {
              using TL = std::remove_pointer_t<decltype(lptr)>;
              std::visit(
                [&](auto *rptr) {
                  using TR = std::remove_pointer_t<decltype(rptr)>;
                  cytnx::linalg_internal::AddInternalImpl<TL, TR>(
                    out._impl->storage()._impl, Lt._impl->storage()._impl,
                    Rt._impl->storage()._impl, Lt._impl->storage()._impl->size(), Lt._impl->shape(),
                    Lt._impl->invmapper(), Rt._impl->invmapper());
                },
                Rt.ptr());
            },
            Lt.ptr());
        } else {
  #ifdef UNI_GPU
          checkCudaErrors(cudaSetDevice(Rt.device()));
          linalg_internal::cuAdd_dispatch(out._impl->storage()._impl, Lt._impl->storage()._impl,
                                          Rt._impl->storage()._impl,
                                          Lt._impl->storage()._impl->size(), Lt._impl->shape(),
                                          Lt._impl->invmapper(), Rt._impl->invmapper());
  #else
          cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
  #endif
        }
      }
      return out;
    }

    //-----------------------------------------------------------------------------------
    template <class T>
    Tensor Add(const T &lc, const Tensor &Rt) {
      static_assert(!std::is_same_v<T, Scalar>,
                    "Add<Scalar>(const Scalar&, const Tensor&) is specialized separately.");
      return detail::AddScalarTensorImpl(lc, Rt);
    }

    template <>
    Tensor Add<Scalar>(const Scalar &lc, const Tensor &Rt) {
      Storage Cnst(1, lc.dtype());
      Cnst.set_item(0, lc);

      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Rt._impl->storage().size(),
                detail::SelectAddOutputType(lc.dtype(), Rt.dtype(), Rt.device()), Rt.device());

      if (Rt.device() == Device.cpu) {
        std::visit(
          [&](auto *lptr) {
            using TL = std::remove_pointer_t<decltype(lptr)>;
            std::visit(
              [&](auto *rptr) {
                using TR = std::remove_pointer_t<decltype(rptr)>;
                cytnx::linalg_internal::AddInternalImpl<TL, TR>(
                  out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
                  Rt._impl->storage()._impl->size(), {}, {}, {});
              },
              Rt.ptr());
          },
          Tensor::from_storage(Cnst).ptr());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::cuAdd_dispatch(out._impl->storage()._impl, Cnst._impl,
                                        Rt._impl->storage()._impl,
                                        Rt._impl->storage()._impl->size(), {}, {}, {});
  #else
        cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }
      return out;
    }

    //-----------------------------------------------------------------------------------
    template <class T>
    Tensor Add(const Tensor &Lt, const T &rc) {
      return Add(rc, Lt);
    }

    template Tensor Add<cytnx_complex128>(const cytnx_complex128 &lc, const Tensor &Rt);
    template Tensor Add<cytnx_complex64>(const cytnx_complex64 &lc, const Tensor &Rt);
    template Tensor Add<cytnx_double>(const cytnx_double &lc, const Tensor &Rt);
    template Tensor Add<cytnx_float>(const cytnx_float &lc, const Tensor &Rt);
    template Tensor Add<cytnx_int64>(const cytnx_int64 &lc, const Tensor &Rt);
    template Tensor Add<cytnx_uint64>(const cytnx_uint64 &lc, const Tensor &Rt);
    template Tensor Add<cytnx_int32>(const cytnx_int32 &lc, const Tensor &Rt);
    template Tensor Add<cytnx_uint32>(const cytnx_uint32 &lc, const Tensor &Rt);
    template Tensor Add<cytnx_int16>(const cytnx_int16 &lc, const Tensor &Rt);
    template Tensor Add<cytnx_uint16>(const cytnx_uint16 &lc, const Tensor &Rt);
    template Tensor Add<cytnx_bool>(const cytnx_bool &lc, const Tensor &Rt);
    template Tensor Add<Scalar>(const Scalar &lc, const Tensor &Rt);

    template Tensor Add<cytnx_complex128>(const Tensor &Lt, const cytnx_complex128 &rc);
    template Tensor Add<cytnx_complex64>(const Tensor &Lt, const cytnx_complex64 &rc);
    template Tensor Add<cytnx_double>(const Tensor &Lt, const cytnx_double &rc);
    template Tensor Add<cytnx_float>(const Tensor &Lt, const cytnx_float &rc);
    template Tensor Add<cytnx_int64>(const Tensor &Lt, const cytnx_int64 &rc);
    template Tensor Add<cytnx_uint64>(const Tensor &Lt, const cytnx_uint64 &rc);
    template Tensor Add<cytnx_int32>(const Tensor &Lt, const cytnx_int32 &rc);
    template Tensor Add<cytnx_uint32>(const Tensor &Lt, const cytnx_uint32 &rc);
    template Tensor Add<cytnx_int16>(const Tensor &Lt, const cytnx_int16 &rc);
    template Tensor Add<cytnx_uint16>(const Tensor &Lt, const cytnx_uint16 &rc);
    template Tensor Add<cytnx_bool>(const Tensor &Lt, const cytnx_bool &rc);
    template Tensor Add<Scalar>(const Tensor &Lt, const Scalar &rc);

  }  // namespace linalg
  //===================
  // operators:
  //===================

  //-------------------------------------------
  Tensor operator+(const Tensor &Lt, const Tensor &Rt) { return linalg::Add(Lt, Rt); }

  template <>
  Tensor operator+<cytnx_complex128>(const cytnx_complex128 &lc, const Tensor &Rt) {
    return linalg::Add(lc, Rt);
  }
  template <>
  Tensor operator+<cytnx_complex64>(const cytnx_complex64 &lc, const Tensor &Rt) {
    return linalg::Add(lc, Rt);
  }
  template <>
  Tensor operator+<cytnx_double>(const cytnx_double &lc, const Tensor &Rt) {
    return linalg::Add(lc, Rt);
  }
  template <>
  Tensor operator+<cytnx_float>(const cytnx_float &lc, const Tensor &Rt) {
    return linalg::Add(lc, Rt);
  }
  template <>
  Tensor operator+<cytnx_int64>(const cytnx_int64 &lc, const Tensor &Rt) {
    return linalg::Add(lc, Rt);
  }
  template <>
  Tensor operator+<cytnx_uint64>(const cytnx_uint64 &lc, const Tensor &Rt) {
    return linalg::Add(lc, Rt);
  }
  template <>
  Tensor operator+<cytnx_int32>(const cytnx_int32 &lc, const Tensor &Rt) {
    return linalg::Add(lc, Rt);
  }
  template <>
  Tensor operator+<cytnx_uint32>(const cytnx_uint32 &lc, const Tensor &Rt) {
    return linalg::Add(lc, Rt);
  }
  template <>
  Tensor operator+<cytnx_int16>(const cytnx_int16 &lc, const Tensor &Rt) {
    return linalg::Add(lc, Rt);
  }
  template <>
  Tensor operator+<cytnx_uint16>(const cytnx_uint16 &lc, const Tensor &Rt) {
    return linalg::Add(lc, Rt);
  }
  template <>
  Tensor operator+<cytnx_bool>(const cytnx_bool &lc, const Tensor &Rt) {
    return linalg::Add(lc, Rt);
  }
  template <>
  Tensor operator+<Scalar>(const Scalar &lc, const Tensor &Rt) {
    return linalg::Add(lc, Rt);
  }
  template <>
  Tensor operator+<Tensor::Tproxy>(const Tensor::Tproxy &lc, const Tensor &Rt) {
    return Tensor(lc) + Rt;
  }
  template <>
  Tensor operator+<Scalar::Sproxy>(const Scalar::Sproxy &lc, const Tensor &Rt) {
    return Scalar(lc) + Rt;
  }

  template <>
  Tensor operator+<cytnx_complex128>(const Tensor &Lt, const cytnx_complex128 &rc) {
    return linalg::Add(Lt, rc);
  }
  template <>
  Tensor operator+<cytnx_complex64>(const Tensor &Lt, const cytnx_complex64 &rc) {
    return linalg::Add(Lt, rc);
  }
  template <>
  Tensor operator+<cytnx_double>(const Tensor &Lt, const cytnx_double &rc) {
    return linalg::Add(Lt, rc);
  }
  template <>
  Tensor operator+<cytnx_float>(const Tensor &Lt, const cytnx_float &rc) {
    return linalg::Add(Lt, rc);
  }
  template <>
  Tensor operator+<cytnx_int64>(const Tensor &Lt, const cytnx_int64 &rc) {
    return linalg::Add(Lt, rc);
  }
  template <>
  Tensor operator+<cytnx_uint64>(const Tensor &Lt, const cytnx_uint64 &rc) {
    return linalg::Add(Lt, rc);
  }
  template <>
  Tensor operator+<cytnx_int32>(const Tensor &Lt, const cytnx_int32 &rc) {
    return linalg::Add(Lt, rc);
  }
  template <>
  Tensor operator+<cytnx_uint32>(const Tensor &Lt, const cytnx_uint32 &rc) {
    return linalg::Add(Lt, rc);
  }
  template <>
  Tensor operator+<cytnx_int16>(const Tensor &Lt, const cytnx_int16 &rc) {
    return linalg::Add(Lt, rc);
  }
  template <>
  Tensor operator+<cytnx_uint16>(const Tensor &Lt, const cytnx_uint16 &rc) {
    return linalg::Add(Lt, rc);
  }
  template <>
  Tensor operator+<cytnx_bool>(const Tensor &Lt, const cytnx_bool &rc) {
    return linalg::Add(Lt, rc);
  }
  template <>
  Tensor operator+<Scalar>(const Tensor &Lt, const Scalar &rc) {
    return linalg::Add(Lt, rc);
  }

}  // namespace cytnx

namespace cytnx {
  namespace linalg {
    //============================================
    // cytnx::UniTensor
    //============================================

    cytnx::UniTensor Add(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt) {
      UniTensor out;
      if (Lt.dtype() > Rt.dtype()) {
        out = Rt.clone();
        out.Add_(Lt);
      } else {
        out = Lt.clone();
        out.Add_(Rt);
      }
      out.set_labels(vec_range<std::string>(Lt.rank()));
      out.set_name("");

      return out;
    }

    //-----------------------------------------------------------------------------------

    template <class T>
    cytnx::UniTensor Add(const T &lc, const cytnx::UniTensor &Rt) {
      // cytnx_error_msg(Rt.is_tag(),"[ERROR] cannot perform arithmetic on tagged
      // unitensor.%s","\n");

      UniTensor out;
      if (Scalar(lc).dtype() < Rt.dtype()) {
        out = Rt.astype(Scalar(lc).dtype());
        out.Add_(lc);
      } else {
        out = Rt.clone();
        out.Add_(lc);
      }
      out.set_name("");

      return out;
    }

    template cytnx::UniTensor Add<cytnx_complex128>(const cytnx_complex128 &lc,
                                                    const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Add<cytnx_complex64>(const cytnx_complex64 &lc,
                                                   const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Add<cytnx_double>(const cytnx_double &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Add<cytnx_float>(const cytnx_float &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Add<cytnx_int64>(const cytnx_int64 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Add<cytnx_uint64>(const cytnx_uint64 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Add<cytnx_int32>(const cytnx_int32 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Add<cytnx_uint32>(const cytnx_uint32 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Add<cytnx_int16>(const cytnx_int16 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Add<cytnx_uint16>(const cytnx_uint16 &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Add<cytnx_bool>(const cytnx_bool &lc, const cytnx::UniTensor &Rt);
    template cytnx::UniTensor Add<Scalar>(const Scalar &lc, const cytnx::UniTensor &Rt);

    //-----------------------------------------------------------------------------------
    template <>
    cytnx::UniTensor Add<cytnx_complex128>(const cytnx::UniTensor &Lt, const cytnx_complex128 &rc) {
      return Add(rc, Lt);
    }
    template <>
    cytnx::UniTensor Add<cytnx_complex64>(const cytnx::UniTensor &Lt, const cytnx_complex64 &rc) {
      return Add(rc, Lt);
    }
    template <>
    cytnx::UniTensor Add<cytnx_double>(const cytnx::UniTensor &Lt, const cytnx_double &rc) {
      return Add(rc, Lt);
    }
    template <>
    cytnx::UniTensor Add<cytnx_float>(const cytnx::UniTensor &Lt, const cytnx_float &rc) {
      return Add(rc, Lt);
    }
    template <>
    cytnx::UniTensor Add<cytnx_int64>(const cytnx::UniTensor &Lt, const cytnx_int64 &rc) {
      return Add(rc, Lt);
    }
    template <>
    cytnx::UniTensor Add<cytnx_uint64>(const cytnx::UniTensor &Lt, const cytnx_uint64 &rc) {
      return Add(rc, Lt);
    }
    template <>
    cytnx::UniTensor Add<cytnx_int32>(const cytnx::UniTensor &Lt, const cytnx_int32 &rc) {
      return Add(rc, Lt);
    }
    template <>
    cytnx::UniTensor Add<cytnx_uint32>(const cytnx::UniTensor &Lt, const cytnx_uint32 &rc) {
      return Add(rc, Lt);
    }
    template <>
    cytnx::UniTensor Add<cytnx_int16>(const cytnx::UniTensor &Lt, const cytnx_int16 &rc) {
      return Add(rc, Lt);
    }
    template <>
    cytnx::UniTensor Add<cytnx_uint16>(const cytnx::UniTensor &Lt, const cytnx_uint16 &rc) {
      return Add(rc, Lt);
    }
    template <>
    cytnx::UniTensor Add<cytnx_bool>(const cytnx::UniTensor &Lt, const cytnx_bool &rc) {
      return Add(rc, Lt);
    }

    template <>
    cytnx::UniTensor Add<Scalar>(const cytnx::UniTensor &Lt, const Scalar &rc) {
      return Add(rc, Lt);
    }

  }  // namespace linalg

  //-------------------------------------------
  cytnx::UniTensor operator+(const cytnx::UniTensor &Lt, const cytnx::UniTensor &Rt) {
    return cytnx::linalg::Add(Lt, Rt);
  }

  template <class T>
  cytnx::UniTensor operator+(const T &lc, const cytnx::UniTensor &Rt) {
    return cytnx::linalg::Add(lc, Rt);
  }
  template <>
  cytnx::UniTensor operator+<Scalar::Sproxy>(const Scalar::Sproxy &lc, const cytnx::UniTensor &Rt) {
    return cytnx::linalg::Add(Scalar(lc), Rt);
  }

  template cytnx::UniTensor operator+
    <cytnx_complex128>(const cytnx_complex128 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator+
    <cytnx_complex64>(const cytnx_complex64 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator+
    <cytnx_double>(const cytnx_double &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator+
    <cytnx_float>(const cytnx_float &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator+
    <cytnx_int64>(const cytnx_int64 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator+
    <cytnx_uint64>(const cytnx_uint64 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator+
    <cytnx_int32>(const cytnx_int32 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator+
    <cytnx_uint32>(const cytnx_uint32 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator+
    <cytnx_int16>(const cytnx_int16 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator+
    <cytnx_uint16>(const cytnx_uint16 &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator+<cytnx_bool>(const cytnx_bool &lc, const cytnx::UniTensor &Rt);
  template cytnx::UniTensor operator+<Scalar>(const Scalar &lc, const cytnx::UniTensor &Rt);

  template <>
  cytnx::UniTensor operator+
    <cytnx_complex128>(const cytnx::UniTensor &Lt, const cytnx_complex128 &rc) {
    return cytnx::linalg::Add(Lt, rc);
  }
  template <>
  cytnx::UniTensor operator+
    <cytnx_complex64>(const cytnx::UniTensor &Lt, const cytnx_complex64 &rc) {
    return cytnx::linalg::Add(Lt, rc);
  }
  template <>
  cytnx::UniTensor operator+<cytnx_double>(const cytnx::UniTensor &Lt, const cytnx_double &rc) {
    return cytnx::linalg::Add(Lt, rc);
  }
  template <>
  cytnx::UniTensor operator+<cytnx_float>(const cytnx::UniTensor &Lt, const cytnx_float &rc) {
    return cytnx::linalg::Add(Lt, rc);
  }
  template <>
  cytnx::UniTensor operator+<cytnx_int64>(const cytnx::UniTensor &Lt, const cytnx_int64 &rc) {
    return cytnx::linalg::Add(Lt, rc);
  }
  template <>
  cytnx::UniTensor operator+<cytnx_uint64>(const cytnx::UniTensor &Lt, const cytnx_uint64 &rc) {
    return cytnx::linalg::Add(Lt, rc);
  }
  template <>
  cytnx::UniTensor operator+<cytnx_int32>(const cytnx::UniTensor &Lt, const cytnx_int32 &rc) {
    return cytnx::linalg::Add(Lt, rc);
  }
  template <>
  cytnx::UniTensor operator+<cytnx_uint32>(const cytnx::UniTensor &Lt, const cytnx_uint32 &rc) {
    return cytnx::linalg::Add(Lt, rc);
  }
  template <>
  cytnx::UniTensor operator+<cytnx_int16>(const cytnx::UniTensor &Lt, const cytnx_int16 &rc) {
    return cytnx::linalg::Add(Lt, rc);
  }
  template <>
  cytnx::UniTensor operator+<cytnx_uint16>(const cytnx::UniTensor &Lt, const cytnx_uint16 &rc) {
    return cytnx::linalg::Add(Lt, rc);
  }
  template <>
  cytnx::UniTensor operator+<cytnx_bool>(const cytnx::UniTensor &Lt, const cytnx_bool &rc) {
    return cytnx::linalg::Add(Lt, rc);
  }

  template <>
  cytnx::UniTensor operator+<Scalar>(const cytnx::UniTensor &Lt, const Scalar &rc) {
    return cytnx::linalg::Add(Lt, rc);
  }
  template <>
  cytnx::UniTensor operator+<Scalar::Sproxy>(const cytnx::UniTensor &Lt, const Scalar::Sproxy &rc) {
    return cytnx::linalg::Add(Lt, Scalar(rc));
  }

}  // namespace cytnx

#endif  // BACKEND_TORCH
