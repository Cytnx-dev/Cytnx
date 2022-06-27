#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"

namespace cytnx {
  namespace linalg {
    Tensor Add(const Tensor &Lt, const Tensor &Rt) {
      cytnx_error_msg(Lt.device() != Rt.device(),
                      "[Add] error, two tensor cannot on different devices.%s", "\n");

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
                        "[Add] error, the two tensor does not have the same shape. Lt rank: [%d] "
                        "Rt rank: [%d] %s",
                        Lt.shape().size(), Rt.shape().size(), "\n");
        out.Init(Lt.shape(), Lt.dtype() < Rt.dtype() ? Lt.dtype() : Rt.dtype(), Lt.device());
      }

      // if contiguous, then no need to calculate the mappers
      if ((Lt.is_contiguous() && Rt.is_contiguous()) || icnst) {
        // contiguous section.
        if (Lt.device() == Device.cpu) {
          cytnx::linalg_internal::lii.Ari_ii[Lt.dtype()][Rt.dtype()](
            out._impl->storage()._impl, Lt._impl->storage()._impl, Rt._impl->storage()._impl,
            out._impl->storage()._impl->size(), {}, {}, {}, 0);
        } else {
#ifdef UNI_GPU
          checkCudaErrors(cudaSetDevice(Rt.device()));
          linalg_internal::lii.cuAri_ii[Lt.dtype()][Rt.dtype()](
            out._impl->storage()._impl, Lt._impl->storage()._impl, Rt._impl->storage()._impl,
            out._impl->storage()._impl->size(), {}, {}, {}, 0);
#else
          cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
#endif
        }
      } else {
        // non-contiguous section
        if (Lt.device() == Device.cpu) {
          linalg_internal::lii.Ari_ii[Lt.dtype()][Rt.dtype()](
            out._impl->storage()._impl, Lt._impl->storage()._impl, Rt._impl->storage()._impl,
            Lt._impl->storage()._impl->size(), Lt._impl->shape(), Lt._impl->invmapper(),
            Rt._impl->invmapper(), 0);
        } else {
#ifdef UNI_GPU
          cytnx_error_msg(true,
                          "[Add][on GPU/CUDA] error two tensors must be contiguous. Call "
                          "Contiguous_() or Contiguous() first%s",
                          "\n");
#else
          cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
#endif
        }
      }
      return out;
    }

    //-----------------------------------------------------------------------------------
    template <>
    Tensor Add<cytnx_complex128>(const cytnx_complex128 &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.ComplexDouble);
      Cnst.at<cytnx_complex128>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Rt._impl->storage().size(), Type.ComplexDouble, Rt.device());
      // Tensor out(Rt.shape(),Type.ComplexDouble,Rt.device());

      if (Rt.device() == Device.cpu) {
        linalg_internal::lii.Ari_ii[Type.ComplexDouble][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::lii.cuAri_ii[Type.ComplexDouble][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
#else
        cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Add<cytnx_complex64>(const cytnx_complex64 &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.ComplexFloat);
      Cnst.at<cytnx_complex64>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Rt._impl->storage().size(),
                Type.ComplexFloat < Rt.dtype() ? Type.ComplexFloat : Rt.dtype(), Rt.device());
      // Tensor out(Rt.shape(),Type.ComplexFloat <
      // Rt.dtype()?Type.ComplexFloat:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        linalg_internal::lii.Ari_ii[Type.ComplexFloat][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::lii.cuAri_ii[Type.ComplexFloat][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
#else
        cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Add<cytnx_double>(const cytnx_double &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Double);
      Cnst.at<cytnx_double>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.Double < Rt.dtype() ? Type.Double : Rt.dtype(),
                Rt.device());
      // Tensor out(Rt.shape(),Type.Double < Rt.dtype()?Type.Double:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        linalg_internal::lii.Ari_ii[Type.Double][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::lii.cuAri_ii[Type.Double][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
#else
        cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Add<cytnx_float>(const cytnx_float &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Float);
      Cnst.at<cytnx_float>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(
        Rt._impl->storage().size(), Type.Float < Rt.dtype() ? Type.Float : Rt.dtype(), Rt.device());
      // Tensor out(Rt.shape(),Type.Float < Rt.dtype()?Type.Float:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        linalg_internal::lii.Ari_ii[Type.Float][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::lii.cuAri_ii[Type.Float][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
#else
        cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Add<cytnx_int64>(const cytnx_int64 &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Int64);
      Cnst.at<cytnx_int64>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(
        Rt._impl->storage().size(), Type.Int64 < Rt.dtype() ? Type.Int64 : Rt.dtype(), Rt.device());
      // Tensor out(Rt.shape(),Type.Int64 < Rt.dtype()?Type.Int64:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        linalg_internal::lii.Ari_ii[Type.Int64][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::lii.cuAri_ii[Type.Int64][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
#else
        cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Add<cytnx_uint64>(const cytnx_uint64 &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Uint64);
      Cnst.at<cytnx_uint64>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.Uint64 < Rt.dtype() ? Type.Uint64 : Rt.dtype(),
                Rt.device());
      // Tensor out(Rt.shape(),Type.Uint64 < Rt.dtype()?Type.Uint64:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        linalg_internal::lii.Ari_ii[Type.Uint64][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::lii.cuAri_ii[Type.Uint64][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
#else
        cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Add<cytnx_int32>(const cytnx_int32 &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Int32);
      Cnst.at<cytnx_int32>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(
        Rt._impl->storage().size(), Type.Int32 < Rt.dtype() ? Type.Int32 : Rt.dtype(), Rt.device());
      // Tensor out(Rt.shape(),Type.Int32 < Rt.dtype()?Type.Int32:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        linalg_internal::lii.Ari_ii[Type.Int32][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::lii.cuAri_ii[Type.Int32][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
#else
        cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Add<cytnx_uint32>(const cytnx_uint32 &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Uint32);
      Cnst.at<cytnx_uint32>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.Uint32 < Rt.dtype() ? Type.Uint32 : Rt.dtype(),
                Rt.device());
      // Tensor out(Rt.shape(),Type.Uint32 < Rt.dtype()?Type.Uint32:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        linalg_internal::lii.Ari_ii[Type.Uint32][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::lii.cuAri_ii[Type.Uint32][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
#else
        cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Add<cytnx_int16>(const cytnx_int16 &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Int16);
      Cnst.at<cytnx_int16>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(
        Rt._impl->storage().size(), Type.Int16 < Rt.dtype() ? Type.Int16 : Rt.dtype(), Rt.device());
      // Tensor out(Rt.shape(),Type.Int16 < Rt.dtype()?Type.Int16:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        linalg_internal::lii.Ari_ii[Type.Int16][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::lii.cuAri_ii[Type.Int16][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
#else
        cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Add<cytnx_uint16>(const cytnx_uint16 &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Uint16);
      Cnst.at<cytnx_uint16>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() =
        Storage(Rt._impl->storage().size(), Type.Uint16 < Rt.dtype() ? Type.Uint16 : Rt.dtype(),
                Rt.device());
      // Tensor out(Rt.shape(),Type.Uint16 < Rt.dtype()?Type.Uint16:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        linalg_internal::lii.Ari_ii[Type.Uint16][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::lii.cuAri_ii[Type.Uint16][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
#else
        cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Add<cytnx_bool>(const cytnx_bool &lc, const Tensor &Rt) {
      Storage Cnst(1, Type.Bool);
      Cnst.at<cytnx_bool>(0) = lc;
      Tensor out;
      out._impl = Rt._impl->_clone_meta_only();
      out._impl->storage() = Storage(Rt._impl->storage().size(),
                                     Type.Bool < Rt.dtype() ? Type.Bool : Rt.dtype(), Rt.device());
      // Tensor out(Rt.shape(),Type.Bool < Rt.dtype()?Type.Bool:Rt.dtype(),Rt.device());

      if (Rt.device() == Device.cpu) {
        linalg_internal::lii.Ari_ii[Type.Bool][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::lii.cuAri_ii[Type.Bool][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
#else
        cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }

    template <>
    Tensor Add<Scalar>(const Scalar &lc, const Tensor &Rt) {
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
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Rt.device()));
        linalg_internal::lii.cuAri_ii[lc.dtype()][Rt.dtype()](
          out._impl->storage()._impl, Cnst._impl, Rt._impl->storage()._impl,
          Rt._impl->storage()._impl->size(), {}, {}, {}, 0);
#else
        cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      // swap back to prevent also free by recycle mech.
      Cnst._impl->Mem = nullptr;

      return out;
    }

    //-----------------------------------------------------------------------------------
    template <>
    Tensor Add<cytnx_complex128>(const Tensor &Lt, const cytnx_complex128 &rc) {
      return Add(rc, Lt);
    }
    template <>
    Tensor Add<cytnx_complex64>(const Tensor &Lt, const cytnx_complex64 &rc) {
      return Add(rc, Lt);
    }
    template <>
    Tensor Add<cytnx_double>(const Tensor &Lt, const cytnx_double &rc) {
      return Add(rc, Lt);
    }
    template <>
    Tensor Add<cytnx_float>(const Tensor &Lt, const cytnx_float &rc) {
      return Add(rc, Lt);
    }
    template <>
    Tensor Add<cytnx_int64>(const Tensor &Lt, const cytnx_int64 &rc) {
      return Add(rc, Lt);
    }
    template <>
    Tensor Add<cytnx_uint64>(const Tensor &Lt, const cytnx_uint64 &rc) {
      return Add(rc, Lt);
    }
    template <>
    Tensor Add<cytnx_int32>(const Tensor &Lt, const cytnx_int32 &rc) {
      return Add(rc, Lt);
    }
    template <>
    Tensor Add<cytnx_uint32>(const Tensor &Lt, const cytnx_uint32 &rc) {
      return Add(rc, Lt);
    }
    template <>
    Tensor Add<cytnx_int16>(const Tensor &Lt, const cytnx_int16 &rc) {
      return Add(rc, Lt);
    }
    template <>
    Tensor Add<cytnx_uint16>(const Tensor &Lt, const cytnx_uint16 &rc) {
      return Add(rc, Lt);
    }
    template <>
    Tensor Add<cytnx_bool>(const Tensor &Lt, const cytnx_bool &rc) {
      return Add(rc, Lt);
    }
    template <>
    Tensor Add<Scalar>(const Tensor &Lt, const Scalar &rc) {
      return Add(rc, Lt);
    }

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
      cytnx_error_msg(Lt.is_tag(), "[ERROR] cannot perform arithmetic on tagged unitensor L.%s",
                      "\n");
      cytnx_error_msg(Rt.is_tag(), "[ERROR] cannot perform arithmetic on tagged unitensor R.%s",
                      "\n");

      UniTensor out = Lt.clone();
      out.set_labels(vec_range<cytnx_int64>(Lt.rank()));
      out.set_name("");

      out.Add_(Rt);

      return out;
    }

    //-----------------------------------------------------------------------------------

    template <class T>
    cytnx::UniTensor Add(const T &lc, const cytnx::UniTensor &Rt) {
      // cytnx_error_msg(Rt.is_tag(),"[ERROR] cannot perform arithmetic on tagged
      // unitensor.%s","\n");

      UniTensor out = Rt.clone();
      // out.set_labels(vec_range<cytnx_int64>(Rt.rank()));
      out.set_name("");

      out.Add_(lc);
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
