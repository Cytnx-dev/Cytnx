#include "linalg.hpp"

#include "Tensor.hpp"
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "../backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {

    void iSub(Tensor &Lt, const Tensor &Rt) {
      cytnx_error_msg(Lt.device() != Rt.device(),
                      "[iSub] error, two tensor cannot on different devices.%s", "\n");

      if (!(Rt.shape().size() == 1 && Rt.shape()[0] == 1)) {
        cytnx_error_msg(Lt.shape() != Rt.shape(),
                        "[iSub] error, the two tensor does not have the same shape. Lt rank: [%d] "
                        "Rt rank: [%d] %s",
                        Lt.shape().size(), Rt.shape().size(), "\n");
      }

      Tensor R;
      if (Lt._impl->storage()._impl == Rt._impl->storage()._impl) {
        R = Rt.clone();
      } else {
        R = Rt;
      }

      Storage nulls;
      // if contiguous, then no need to calculate the mappers
      if ((Lt.is_contiguous() && Rt.is_contiguous())) {
        // contiguous section.
        if (Lt.device() == Device.cpu) {
          cytnx::linalg_internal::lii.iAri_ii[Lt.dtype()][Rt.dtype()](
            nulls._impl, Lt._impl->storage()._impl, R._impl->storage()._impl,
            Lt._impl->storage()._impl->size(), {}, {}, {}, 2);
        } else {
  #ifdef UNI_GPU
          checkCudaErrors(cudaSetDevice(Rt.device()));
          Tensor tmpo;
          if (Lt.dtype() <= Rt.dtype())
            tmpo = Lt;
          else
            tmpo = Lt.clone();
          linalg_internal::lii.cuAri_ii[Lt.dtype()][Rt.dtype()](
            tmpo._impl->storage()._impl, Lt._impl->storage()._impl, R._impl->storage()._impl,
            Lt._impl->storage()._impl->size(), {}, {}, {}, 2);
          // cytnx_error_msg(true, "[Developing] iAdd for GPU%s", "\n");

          if (Lt.dtype() > Rt.dtype()) Lt = tmpo;

  #else
          cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
  #endif
        }
      } else {
        // non-contiguous section
        if (Lt.device() == Device.cpu) {
          linalg_internal::lii.iAri_ii[Lt.dtype()][Rt.dtype()](
            nulls._impl, Lt._impl->storage()._impl, R._impl->storage()._impl,
            Lt._impl->storage()._impl->size(), Lt._impl->shape(), Lt._impl->invmapper(),
            Rt._impl->invmapper(), 2);
        } else {
  #ifdef UNI_GPU
          checkCudaErrors(cudaSetDevice(Rt.device()));
          Tensor tmpo;
          if (Lt.dtype() <= Rt.dtype())
            tmpo = Lt;
          else
            tmpo = Lt.clone();
          linalg_internal::lii.cuAri_ii[Lt.dtype()][Rt.dtype()](
            tmpo._impl->storage()._impl, Lt._impl->storage()._impl, R._impl->storage()._impl,
            Lt._impl->storage()._impl->size(), Lt._impl->shape(), Lt._impl->invmapper(),
            Rt._impl->invmapper(), 2);
          if (Lt.dtype() > Rt.dtype()) Lt = tmpo;

  #else
          cytnx_error_msg(true, "[Sub] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
  #endif
        }
      }
    }

  }  // namespace linalg
}  // namespace cytnx
#endif
