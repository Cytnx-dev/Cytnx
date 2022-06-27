#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include <iostream>
#include "Tensor.hpp"
namespace cytnx {
  namespace linalg {
    Tensor Det(const Tensor &Tl) {
      cytnx_error_msg(Tl.shape().size() != 2,
                      "[Det] error, tensor Tl , Det can only operate on rank-2 Tensor.%s", "\n");
      cytnx_error_msg(Tl.shape()[0] != Tl.shape()[1],
                      "[Det] error, tensor Tl , Det can only operate on NxN Tensor.%s", "\n");

      // check type:
      Tensor _tl = Tl.contiguous();
      Tensor out;

      if (Tl.dtype() > 4) {
        // do conversion:
        _tl = _tl.astype(Type.Double);
        out.Init({1}, Type.Double, _tl.device());
      } else {
        out.Init({1}, _tl.dtype(), _tl.device());
      }

      if (Tl.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Det_ii[_tl.dtype()](out._impl->storage()._impl->Mem,
                                                        _tl._impl->storage()._impl, Tl.shape()[0]);

        return out;

      } else {
#ifdef UNI_GPU
        cytnx_error_msg(true, "[Det] on GPU Developing!%s", "\n");
        // checkCudaErrors(cudaSetDevice(Tl.device()));
        // cytnx::linalg_internal::lii.cuNorm_ii[_tl.dtype()](out._impl->storage()._impl->Mem,
        //                                         _tl._impl->storage()._impl, Tl.shape()[0]);

        return out;
#else
        cytnx_error_msg(true, "[Det] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return Tensor();
#endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx
