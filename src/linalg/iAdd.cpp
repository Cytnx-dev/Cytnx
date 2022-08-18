#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"

namespace cytnx {
  namespace linalg {

    void iAdd(Tensor &Lt, const Tensor &Rt) {
      cytnx_error_msg(Lt.device() != Rt.device(),
                      "[iAdd] error, two tensor cannot on different devices.%s", "\n");
      if (!(Rt.shape().size() == 1 && Rt.shape()[0] == 1)) {
        cytnx_error_msg(Lt.shape() != Rt.shape(),
                        "[iAdd] error, the two tensor does not have the same shape. Lt rank: [%d] "
                        "Rt rank: [%d] %s",
                        Lt.shape().size(), Rt.shape().size(), "\n");
      }

      // std::cout << "iadd entry" << std::endl;
      Storage nulls;

      Tensor R;
      if (Lt._impl->storage()._impl == Rt._impl->storage()._impl) {
        R = Rt.clone();
      } else {
        R = Rt;
      }

      // if contiguous, then no need to calculate the mappers
      if ((Lt.is_contiguous() && Rt.is_contiguous())) {
        // contiguous section.
        if (Lt.device() == Device.cpu) {
          cytnx::linalg_internal::lii.iAri_ii[Lt.dtype()][Rt.dtype()](
            nulls._impl, Lt._impl->storage()._impl, R._impl->storage()._impl,
            Lt._impl->storage()._impl->size(), {}, {}, {}, 0);
        } else {
#ifdef UNI_GPU
          checkCudaErrors(cudaSetDevice(Rt.device()));
          // linalg_internal::lii.cuiAri_ii[Lt.dtype()][Rt.dtype()](nullptr,Lt._impl->storage()._impl,Rt._impl->storage()._impl,oLt._impl->storage()._impl->size(),{},{},{},0);
          cytnx_error_msg(true, "[Developing] iAdd for GPU%s", "\n");
#else
          cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
#endif
        }
      } else {
        // non-contiguous section
        if (Lt.device() == Device.cpu) {
          linalg_internal::lii.iAri_ii[Lt.dtype()][Rt.dtype()](
            nulls._impl, Lt._impl->storage()._impl, R._impl->storage()._impl,
            Lt._impl->storage()._impl->size(), Lt._impl->shape(), Lt._impl->invmapper(),
            Rt._impl->invmapper(), 0);
        } else {
#ifdef UNI_GPU
          cytnx_error_msg(true, "[Developing] iAdd for GPU%s", "\n");
#else
          cytnx_error_msg(true, "[Add] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
#endif
        }
      }
    }

  }  // namespace linalg
}  // namespace cytnx
