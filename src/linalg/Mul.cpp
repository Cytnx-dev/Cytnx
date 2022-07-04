#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
namespace cytnx {
  namespace linalg {
    Tensor Mul(const Tensor &Lt, const Tensor &Rt) {
      cytnx_error_msg(Lt.device() != Rt.device(),
                      "[Mul] error, two tensor cannot on different devices.%s", "\n");

      Tensor out;
      bool icnst = false;
      if (Lt.shape().size() == 1 && Lt.shape()[0] == 1) {
        out._impl = Rt._impl->_clone_meta_only();
        out._impl->storage() = Storage(
          Rt.storage().size(), Lt.dtype() < Rt.dtype() ? Lt.dtype() : Rt.dtype(), Rt.device());
        // out.Init(Rt.shape(),Lt.dtype() < Rt.dtype()?Lt.dtype():Rt.dtype(),Lt.device());
        icnst = true;

      } else if (Rt.shape().size() == 1 && Rt.shape()[0] == 1) {
        // out.Init(Lt.shape(),Lt.dtype() < Rt.dtype()?Lt.dtype():Rt.dtype(),Lt.device());
        out._impl = Lt._impl->_clone_meta_only();
        out._impl->storage() = Storage(
          Lt.storage().size(), Lt.dtype() < Rt.dtype() ? Lt.dtype() : Rt.dtype(), Lt.device());
        icnst = true;

      } else {
        cytnx_error_msg(Lt.shape() != Rt.shape(),
                        "[Mul] error, the two tensor does not have the same shape.%s", "\n");
        out.Init(Lt.shape(), Lt.dtype() < Rt.dtype() ? Lt.dtype() : Rt.dtype(), Lt.device());
      }

      if ((Lt.is_contiguous() && Rt.is_contiguous()) || icnst) {
        // contiguous section
        if (Lt.device() == Device.cpu) {
          cytnx::linalg_internal::lii.Ari_ii[Lt.dtype()][Rt.dtype()](
            out._impl->storage()._impl, Lt._impl->storage()._impl, Rt._impl->storage()._impl,
            out._impl->storage()._impl->size(), {}, {}, {}, 1);
        } else {
#ifdef UNI_GPU
          checkCudaErrors(cudaSetDevice(Rt.device()));
          cytnx::linalg_internal::lii.cuAri_ii[Lt.dtype()][Rt.dtype()](
            out._impl->storage()._impl, Lt._impl->storage()._impl, Rt._impl->storage()._impl,
            out._impl->storage()._impl->size(), {}, {}, {}, 1);
#else
          cytnx_error_msg(true, "[Mul] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
#endif
        }
      } else {
        // non-contiguous section
        if (Lt.device() == Device.cpu) {
          cytnx::linalg_internal::lii.Ari_ii[Lt.dtype()][Rt.dtype()](
            out._impl->storage()._impl, Lt._impl->storage()._impl, Rt._impl->storage()._impl,
            Lt._impl->storage()._impl->size(), Lt._impl->shape(), Lt._impl->invmapper(),
            Rt._impl->invmapper(), 1);
        } else {
#ifdef UNI_GPU
          cytnx_error_msg(true,
                          "[Mul][on GPU/CUDA] error two tensors must be contiguous. Call "
                          "Contiguous_() or Contiguous() first%s",
                          "\n");
#else
          cytnx_error_msg(true, "[Mul] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
#endif
        }
      }
      return out;
    }

    //-----------------------------------------------------------------------------------
    template <>
    Tensor Mul<cytnx_complex128>(const cytnx_complex128 &lc, const Tensor &Rt) {
      // std::cout << "entroy" << std::endl;
      Storage Cnst(1, Type.ComplexDouble);
      Cnst.at<cytnx_complex128>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Rt._impl->storage().size(), Type.ComplexDouble, Rt.device());
      // Tensor out(Rt.shape(), Type.ComplexDouble, Rt.device());

      if (Rt.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Ari_ii[Type.ComplexDouble][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::lii.cuAri_ii[Type.ComplexDouble][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
#else
        cytnx_error_msg(true, "[Mul] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Mul<cytnx_complex64>(const cytnx_complex64 &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.ComplexFloat);
      Cnst.at<cytnx_complex64>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Rt._impl->storage().size(),
                Type.ComplexFloat < Rt.dtype() ? Type.ComplexFloat : Rt.dtype(), Rt.device());
      // Tensor out(Rt.shape(), Type.ComplexFloat < Rt.dtype() ? Type.ComplexFloat : Rt.dtype(),
      // Rt.device());

      if (Rt.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Ari_ii[Type.ComplexFloat][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::lii.cuAri_ii[Type.ComplexFloat][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
#else
        cytnx_error_msg(true, "[Mul] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Mul<cytnx_double>(const cytnx_double &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Double);
      Cnst.at<cytnx_double>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.Double < Rt.dtype() ? Type.Double : Rt.dtype(),
                Rt.device());
      // Tensor out(Rt.shape(), Type.Double < Rt.dtype() ? Type.Double : Rt.dtype(), Rt.device());

      if (Rt.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Ari_ii[Type.Double][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::lii.cuAri_ii[Type.Double][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
#else
        cytnx_error_msg(true, "[Mul] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Mul<cytnx_float>(const cytnx_float &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Float);
      Cnst.at<cytnx_float>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(
        Rt._impl->storage().size(), Type.Float < Rt.dtype() ? Type.Float : Rt.dtype(), Rt.device());
      // Tensor out(Rt.shape(), Type.Float < Rt.dtype() ? Type.Float : Rt.dtype(), Rt.device());

      if (Rt.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Ari_ii[Type.Float][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::lii.cuAri_ii[Type.Float][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
#else
        cytnx_error_msg(true, "[Mul] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Mul<cytnx_int64>(const cytnx_int64 &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Int64);
      Cnst.at<cytnx_int64>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(
        Rt._impl->storage().size(), Type.Int64 < Rt.dtype() ? Type.Int64 : Rt.dtype(), Rt.device());
      // Tensor out(Rt.shape(), Type.Int64 < Rt.dtype() ? Type.Int64 : Rt.dtype(), Rt.device());

      if (Rt.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Ari_ii[Type.Int64][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::lii.cuAri_ii[Type.Int64][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
#else
        cytnx_error_msg(true, "[Mul] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Mul<cytnx_uint64>(const cytnx_uint64 &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Uint64);
      Cnst.at<cytnx_uint64>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.Uint64 < Rt.dtype() ? Type.Uint64 : Rt.dtype(),
                Rt.device());
      // Tensor out(Rt.shape(), Type.Uint64 < Rt.dtype() ? Type.Uint64 : Rt.dtype(), Rt.device());

      if (Rt.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Ari_ii[Type.Uint64][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::lii.cuAri_ii[Type.Uint64][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
#else
        cytnx_error_msg(true, "[Mul] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Mul<cytnx_int32>(const cytnx_int32 &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Int32);
      Cnst.at<cytnx_int32>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(
        Rt._impl->storage().size(), Type.Int32 < Rt.dtype() ? Type.Int32 : Rt.dtype(), Rt.device());
      // Tensor out(Rt.shape(), Type.Int32 < Rt.dtype() ? Type.Int32 : Rt.dtype(), Rt.device());

      if (Rt.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Ari_ii[Type.Int32][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::lii.cuAri_ii[Type.Int32][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
#else
        cytnx_error_msg(true, "[Mul] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Mul<cytnx_uint32>(const cytnx_uint32 &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Uint32);
      Cnst.at<cytnx_uint32>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.Uint32 < Rt.dtype() ? Type.Uint32 : Rt.dtype(),
                Rt.device());
      // Tensor out(Rt.shape(), Type.Uint32 < Rt.dtype() ? Type.Uint32 : Rt.dtype(), Rt.device());

      if (Rt.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Ari_ii[Type.Uint32][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::lii.cuAri_ii[Type.Uint32][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
#else
        cytnx_error_msg(true, "[Mul] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Mul<cytnx_int16>(const cytnx_int16 &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Int16);
      Cnst.at<cytnx_int16>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(
        Rt._impl->storage().size(), Type.Int16 < Rt.dtype() ? Type.Int16 : Rt.dtype(), Rt.device());
      // Tensor out(Rt.shape(), Type.Int16 < Rt.dtype() ? Type.Int16 : Rt.dtype(), Rt.device());

      if (Rt.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Ari_ii[Type.Int16][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::lii.cuAri_ii[Type.Int16][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
#else
        cytnx_error_msg(true, "[Mul] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Mul<cytnx_uint16>(const cytnx_uint16 &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Uint16);
      Cnst.at<cytnx_uint16>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.Uint16 < Rt.dtype() ? Type.Uint16 : Rt.dtype(),
                Rt.device());
      // Tensor out(Rt.shape(), Type.Uint16 < Rt.dtype() ? Type.Uint16 : Rt.dtype(), Rt.device());

      if (Rt.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Ari_ii[Type.Uint16][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::lii.cuAri_ii[Type.Uint16][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
#else
        cytnx_error_msg(true, "[Mul] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Mul<cytnx_bool>(const cytnx_bool &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Bool);
      Cnst.at<cytnx_bool>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Rt._impl->storage().size(),
                                     Type.Bool < Rt.dtype() ? Type.Bool : Rt.dtype(), Rt.device());
      // Tensor out(Rt.shape(), Type.Bool < Rt.dtype() ? Type.Bool : Rt.dtype(), Rt.device());

      if (Rt.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Ari_ii[Type.Bool][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        cytnx::linalg_internal::lii.cuAri_ii[Type.Bool][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
#else
        cytnx_error_msg(true, "[Mul] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Mul<Scalar>(const Scalar &lc, const Tensor &Rt) {
      Storage Cnst;  // create a shallow container without allocate. Using base!

      Cnst._impl->Mem = lc._impl->get_raw_address();
      Cnst._impl->len = 1;

      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(
        Rt._impl->storage().size(), lc.dtype() < Rt.dtype() ? lc.dtype() : Rt.dtype(), Rt.device());

      if (Rt.device() == Device.cpu) {
        linalg_internal::lii.Ari_ii[lc.dtype()][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::lii.cuAri_ii[lc.dtype()][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 1);
#else
        cytnx_error_msg(true, "[Mul] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      // swap back to prevent also free by recycle mech.
      Cnst._impl->Mem = nullptr;

      return out;
    }

    //-----------------------------------------------------------------------------------
    template <>
    Tensor Mul<cytnx_complex128>(const Tensor &Lc, const cytnx_complex128 &rc) {
      return Mul(rc, Lc);
    }

    template <>
    Tensor Mul<cytnx_complex64>(const Tensor &Lc, const cytnx_complex64 &rc) {
      return Mul(rc, Lc);
    }

    template <>
    Tensor Mul<cytnx_double>(const Tensor &Lc, const cytnx_double &rc) {
      return Mul(rc, Lc);
    }

    template <>
    Tensor Mul<cytnx_float>(const Tensor &Lc, const cytnx_float &rc) {
      return Mul(rc, Lc);
    }

    template <>
    Tensor Mul<cytnx_int64>(const Tensor &Lc, const cytnx_int64 &rc) {
      return Mul(rc, Lc);
    }

    template <>
    Tensor Mul<cytnx_uint64>(const Tensor &Lc, const cytnx_uint64 &rc) {
      return Mul(rc, Lc);
    }

    template <>
    Tensor Mul<cytnx_int32>(const Tensor &Lc, const cytnx_int32 &rc) {
      return Mul(rc, Lc);
    }

    template <>
    Tensor Mul<cytnx_uint32>(const Tensor &Lc, const cytnx_uint32 &rc) {
      return Mul(rc, Lc);
    }

    template <>
    Tensor Mul<cytnx_int16>(const Tensor &Lc, const cytnx_int16 &rc) {
      return Mul(rc, Lc);
    }

    template <>
    Tensor Mul<cytnx_uint16>(const Tensor &Lc, const cytnx_uint16 &rc) {
      return Mul(rc, Lc);
    }

    template <>
    Tensor Mul<cytnx_bool>(const Tensor &Lc, const cytnx_bool &rc) {
      return Mul(rc, Lc);
    }

    template <>
    Tensor Mul<Scalar>(const Tensor &Lc, const Scalar &rc) {
      return Mul(rc, Lc);
    }

  }  // namespace linalg
  Tensor operator*(const Tensor &Lt, const Tensor &Rt) { return cytnx::linalg::Mul(Lt, Rt); }
  template <>
  Tensor operator*<cytnx_complex128>(const cytnx_complex128 &lc, const Tensor &Rt) {
    return cytnx::linalg::Mul(lc, Rt);
  }
  template <>
  Tensor operator*<cytnx_complex64>(const cytnx_complex64 &lc, const Tensor &Rt) {
    return cytnx::linalg::Mul(lc, Rt);
  }
  template <>
  Tensor operator*<cytnx_double>(const cytnx_double &lc, const Tensor &Rt) {
    return cytnx::linalg::Mul(lc, Rt);
  }
  template <>
  Tensor operator*<cytnx_float>(const cytnx_float &lc, const Tensor &Rt) {
    return cytnx::linalg::Mul(lc, Rt);
  }
  template <>
  Tensor operator*<cytnx_int64>(const cytnx_int64 &lc, const Tensor &Rt) {
    return cytnx::linalg::Mul(lc, Rt);
  }
  template <>
  Tensor operator*<cytnx_uint64>(const cytnx_uint64 &lc, const Tensor &Rt) {
    return cytnx::linalg::Mul(lc, Rt);
  }
  template <>
  Tensor operator*<cytnx_int32>(const cytnx_int32 &lc, const Tensor &Rt) {
    return cytnx::linalg::Mul(lc, Rt);
  }
  template <>
  Tensor operator*<cytnx_uint32>(const cytnx_uint32 &lc, const Tensor &Rt) {
    return cytnx::linalg::Mul(lc, Rt);
  }
  template <>
  Tensor operator*<cytnx_uint16>(const cytnx_uint16 &lc, const Tensor &Rt) {
    return cytnx::linalg::Mul(lc, Rt);
  }
  template <>
  Tensor operator*<cytnx_int16>(const cytnx_int16 &lc, const Tensor &Rt) {
    return cytnx::linalg::Mul(lc, Rt);
  }
  template <>
  Tensor operator*<cytnx_bool>(const cytnx_bool &lc, const Tensor &Rt) {
    return cytnx::linalg::Mul(lc, Rt);
  }
  template <>
  Tensor operator*<Scalar>(const Scalar &lc, const Tensor &Rt) {
    return cytnx::linalg::Mul(lc, Rt);
  }
  template <>
  Tensor operator*<Scalar::Sproxy>(const Scalar::Sproxy &lc, const Tensor &Rt) {
    return cytnx::linalg::Mul(Scalar(lc), Rt);
  }

  template <>
  Tensor operator*<cytnx_complex128>(const Tensor &Lt, const cytnx_complex128 &rc) {
    return cytnx::linalg::Mul(Lt, rc);
  }
  template <>
  Tensor operator*<cytnx_complex64>(const Tensor &Lt, const cytnx_complex64 &rc) {
    return cytnx::linalg::Mul(Lt, rc);
  }
  template <>
  Tensor operator*<cytnx_double>(const Tensor &Lt, const cytnx_double &rc) {
    return cytnx::linalg::Mul(Lt, rc);
  }
  template <>
  Tensor operator*<cytnx_float>(const Tensor &Lt, const cytnx_float &rc) {
    return cytnx::linalg::Mul(Lt, rc);
  }
  template <>
  Tensor operator*<cytnx_int64>(const Tensor &Lt, const cytnx_int64 &rc) {
    return cytnx::linalg::Mul(Lt, rc);
  }
  template <>
  Tensor operator*<cytnx_uint64>(const Tensor &Lt, const cytnx_uint64 &rc) {
    return cytnx::linalg::Mul(Lt, rc);
  }
  template <>
  Tensor operator*<cytnx_int32>(const Tensor &Lt, const cytnx_int32 &rc) {
    return cytnx::linalg::Mul(Lt, rc);
  }
  template <>
  Tensor operator*<cytnx_uint32>(const Tensor &Lt, const cytnx_uint32 &rc) {
    return cytnx::linalg::Mul(Lt, rc);
  }
  template <>
  Tensor operator*<cytnx_uint16>(const Tensor &Lt, const cytnx_uint16 &rc) {
    return cytnx::linalg::Mul(Lt, rc);
  }
  template <>
  Tensor operator*<cytnx_int16>(const Tensor &Lt, const cytnx_int16 &rc) {
    return cytnx::linalg::Mul(Lt, rc);
  }
  template <>
  Tensor operator*<cytnx_bool>(const Tensor &Lt, const cytnx_bool &rc) {
    return cytnx::linalg::Mul(Lt, rc);
  }
  template <>
  Tensor operator*<Scalar>(const Tensor &Lt, const Scalar &rc) {
    return cytnx::linalg::Mul(Lt, rc);
  }
  template <>
  Tensor operator*<Tensor::Tproxy>(const Tensor::Tproxy &lc, const Tensor &Rt) {
    return Tensor(lc) * Rt;
  }

  template <>
  Tensor operator*<cytnx_complex128>(const Tensor &, const cytnx_complex128 &);
  template <>
  Tensor operator*<cytnx_complex64>(const Tensor &, const cytnx_complex64 &);
  template <>
  Tensor operator*<cytnx_double>(const Tensor &, const cytnx_double &);
  template <>
  Tensor operator*<cytnx_float>(const Tensor &, const cytnx_float &);
  template <>
  Tensor operator*<cytnx_int64>(const Tensor &, const cytnx_int64 &);
  template <>
  Tensor operator*<cytnx_uint64>(const Tensor &, const cytnx_uint64 &);
  template <>
  Tensor operator*<cytnx_int32>(const Tensor &, const cytnx_int32 &);
  template <>
  Tensor operator*<cytnx_uint32>(const Tensor &, const cytnx_uint32 &);
  template <>
  Tensor operator*<cytnx_uint16>(const Tensor &, const cytnx_uint16 &);
  template <>
  Tensor operator*<cytnx_int16>(const Tensor &, const cytnx_int16 &);
  template <>
  Tensor operator*<cytnx_bool>(const Tensor &, const cytnx_bool &);
  template <>
  Tensor operator*<Scalar>(const Tensor &, const Scalar &);

  template <>
  Tensor operator*<cytnx_complex128>(const cytnx_complex128 &, const Tensor &);
  template <>
  Tensor operator*<cytnx_complex64>(const cytnx_complex64 &, const Tensor &);
  template <>
  Tensor operator*<cytnx_double>(const cytnx_double &, const Tensor &);
  template <>
  Tensor operator*<cytnx_float>(const cytnx_float &, const Tensor &);
  template <>
  Tensor operator*<cytnx_int64>(const cytnx_int64 &, const Tensor &);
  template <>
  Tensor operator*<cytnx_uint64>(const cytnx_uint64 &, const Tensor &);
  template <>
  Tensor operator*<cytnx_int32>(const cytnx_int32 &, const Tensor &);
  template <>
  Tensor operator*<cytnx_uint32>(const cytnx_uint32 &, const Tensor &);
  template <>
  Tensor operator*<cytnx_uint16>(const cytnx_uint16 &, const Tensor &);
  template <>
  Tensor operator*<cytnx_int16>(const cytnx_int16 &, const Tensor &);
  template <>
  Tensor operator*<cytnx_bool>(const cytnx_bool &, const Tensor &);
  template <>
  Tensor operator*<Scalar>(const Scalar &, const Tensor &);

}  // namespace cytnx

namespace cytnx {
  namespace linalg {

    //============================================
    // UniTensor
    //============================================

    UniTensor Mul(const UniTensor &Lt, const UniTensor &Rt) {
      cytnx_error_msg(Lt.is_tag(), "[ERROR] cannot perform arithmetic on tagged unitensor L.%s",
                      "\n");
      cytnx_error_msg(Rt.is_tag(), "[ERROR] cannot perform arithmetic on tagged unitensor R.%s",
                      "\n");

      UniTensor out = Lt.clone();
      out.set_labels(vec_range<cytnx_int64>(Lt.rank()));
      out.set_name("");

      out.Mul_(Rt);

      return out;
    }

    //-----------------------------------------------------------------------------------

    template <class T>
    UniTensor Mul(const T &lc, const UniTensor &Rt) {
      // cytnx_error_msg(Rt.is_tag(),"[ERROR] cannot perform arithmetic on tagged
      // unitensor.%s","\n");

      UniTensor out = Rt.clone();
      // out.set_labels(vec_range<cytnx_int64>(Rt.rank()));
      out.set_name("");

      out.Mul_(lc);
      return out;
    }

    template UniTensor Mul<cytnx_complex128>(const cytnx_complex128 &lc, const UniTensor &Rt);
    template UniTensor Mul<cytnx_complex64>(const cytnx_complex64 &lc, const UniTensor &Rt);
    template UniTensor Mul<cytnx_double>(const cytnx_double &lc, const UniTensor &Rt);
    template UniTensor Mul<cytnx_float>(const cytnx_float &lc, const UniTensor &Rt);
    template UniTensor Mul<cytnx_int64>(const cytnx_int64 &lc, const UniTensor &Rt);
    template UniTensor Mul<cytnx_uint64>(const cytnx_uint64 &lc, const UniTensor &Rt);
    template UniTensor Mul<cytnx_int32>(const cytnx_int32 &lc, const UniTensor &Rt);
    template UniTensor Mul<cytnx_uint32>(const cytnx_uint32 &lc, const UniTensor &Rt);
    template UniTensor Mul<cytnx_int16>(const cytnx_int16 &lc, const UniTensor &Rt);
    template UniTensor Mul<cytnx_uint16>(const cytnx_uint16 &lc, const UniTensor &Rt);
    template UniTensor Mul<cytnx_bool>(const cytnx_bool &lc, const UniTensor &Rt);
    template UniTensor Mul<Scalar>(const Scalar &lc, const UniTensor &Rt);
    //-----------------------------------------------------------------------------------
    template <>
    UniTensor Mul<cytnx_complex128>(const UniTensor &Lt, const cytnx_complex128 &rc) {
      return Mul(rc, Lt);
    }
    template <>
    UniTensor Mul<cytnx_complex64>(const UniTensor &Lt, const cytnx_complex64 &rc) {
      return Mul(rc, Lt);
    }
    template <>
    UniTensor Mul<cytnx_double>(const UniTensor &Lt, const cytnx_double &rc) {
      return Mul(rc, Lt);
    }
    template <>
    UniTensor Mul<cytnx_float>(const UniTensor &Lt, const cytnx_float &rc) {
      return Mul(rc, Lt);
    }
    template <>
    UniTensor Mul<cytnx_int64>(const UniTensor &Lt, const cytnx_int64 &rc) {
      return Mul(rc, Lt);
    }
    template <>
    UniTensor Mul<cytnx_uint64>(const UniTensor &Lt, const cytnx_uint64 &rc) {
      return Mul(rc, Lt);
    }
    template <>
    UniTensor Mul<cytnx_int32>(const UniTensor &Lt, const cytnx_int32 &rc) {
      return Mul(rc, Lt);
    }
    template <>
    UniTensor Mul<cytnx_uint32>(const UniTensor &Lt, const cytnx_uint32 &rc) {
      return Mul(rc, Lt);
    }
    template <>
    UniTensor Mul<cytnx_int16>(const UniTensor &Lt, const cytnx_int16 &rc) {
      return Mul(rc, Lt);
    }
    template <>
    UniTensor Mul<cytnx_uint16>(const UniTensor &Lt, const cytnx_uint16 &rc) {
      return Mul(rc, Lt);
    }
    template <>
    UniTensor Mul<cytnx_bool>(const UniTensor &Lt, const cytnx_bool &rc) {
      return Mul(rc, Lt);
    }
    template <>
    UniTensor Mul<Scalar>(const UniTensor &Lt, const Scalar &rc) {
      return Mul(rc, Lt);
    }
  }  // namespace linalg

  //-------------------------------------------
  UniTensor operator*(const UniTensor &Lt, const UniTensor &Rt) { return linalg::Mul(Lt, Rt); }

  template <class T>
  UniTensor operator*(const T &lc, const UniTensor &Rt) {
    return linalg::Mul(lc, Rt);
  }
  template <>
  UniTensor operator*<Scalar::Sproxy>(const Scalar::Sproxy &lc, const UniTensor &Rt) {
    return linalg::Mul(Scalar(lc), Rt);
  }

  template UniTensor operator*<cytnx_complex128>(const cytnx_complex128 &lc, const UniTensor &Rt);
  template UniTensor operator*<cytnx_complex64>(const cytnx_complex64 &lc, const UniTensor &Rt);
  template UniTensor operator*<cytnx_double>(const cytnx_double &lc, const UniTensor &Rt);
  template UniTensor operator*<cytnx_float>(const cytnx_float &lc, const UniTensor &Rt);
  template UniTensor operator*<cytnx_int64>(const cytnx_int64 &lc, const UniTensor &Rt);
  template UniTensor operator*<cytnx_uint64>(const cytnx_uint64 &lc, const UniTensor &Rt);
  template UniTensor operator*<cytnx_int32>(const cytnx_int32 &lc, const UniTensor &Rt);
  template UniTensor operator*<cytnx_uint32>(const cytnx_uint32 &lc, const UniTensor &Rt);
  template UniTensor operator*<cytnx_int16>(const cytnx_int16 &lc, const UniTensor &Rt);
  template UniTensor operator*<cytnx_uint16>(const cytnx_uint16 &lc, const UniTensor &Rt);
  template UniTensor operator*<cytnx_bool>(const cytnx_bool &lc, const UniTensor &Rt);
  template UniTensor operator*<Scalar>(const Scalar &lc, const UniTensor &Rt);

  template <>
  UniTensor operator*<cytnx_complex128>(const UniTensor &Lt, const cytnx_complex128 &rc) {
    return linalg::Mul(Lt, rc);
  }
  template <>
  UniTensor operator*<cytnx_complex64>(const UniTensor &Lt, const cytnx_complex64 &rc) {
    return linalg::Mul(Lt, rc);
  }
  template <>
  UniTensor operator*<cytnx_double>(const UniTensor &Lt, const cytnx_double &rc) {
    return linalg::Mul(Lt, rc);
  }
  template <>
  UniTensor operator*<cytnx_float>(const UniTensor &Lt, const cytnx_float &rc) {
    return linalg::Mul(Lt, rc);
  }
  template <>
  UniTensor operator*<cytnx_int64>(const UniTensor &Lt, const cytnx_int64 &rc) {
    return linalg::Mul(Lt, rc);
  }
  template <>
  UniTensor operator*<cytnx_uint64>(const UniTensor &Lt, const cytnx_uint64 &rc) {
    return linalg::Mul(Lt, rc);
  }
  template <>
  UniTensor operator*<cytnx_int32>(const UniTensor &Lt, const cytnx_int32 &rc) {
    return linalg::Mul(Lt, rc);
  }
  template <>
  UniTensor operator*<cytnx_uint32>(const UniTensor &Lt, const cytnx_uint32 &rc) {
    return linalg::Mul(Lt, rc);
  }
  template <>
  UniTensor operator*<cytnx_int16>(const UniTensor &Lt, const cytnx_int16 &rc) {
    return linalg::Mul(Lt, rc);
  }
  template <>
  UniTensor operator*<cytnx_uint16>(const UniTensor &Lt, const cytnx_uint16 &rc) {
    return linalg::Mul(Lt, rc);
  }
  template <>
  UniTensor operator*<cytnx_bool>(const UniTensor &Lt, const cytnx_bool &rc) {
    return linalg::Mul(Lt, rc);
  }
  template <>
  UniTensor operator*<Scalar>(const UniTensor &Lt, const Scalar &rc) {
    return linalg::Mul(Lt, rc);
  }
  template <>
  UniTensor operator*<Scalar::Sproxy>(const UniTensor &Lt, const Scalar::Sproxy &rc) {
    return linalg::Mul(Lt, Scalar(rc));
  }
}  // namespace cytnx
