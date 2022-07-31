#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "utils/utils.hpp"
#include "Tensor.hpp"

namespace cytnx {

  namespace linalg {
    using namespace std;
    Tensor Vectordot(const Tensor &Tl, const Tensor &Tr, const bool &is_conj) {
      // checking:
      cytnx_error_msg(Tl.device() != Tr.device(),
                      "[ERROR] two tensor for Vectordot cannot on different devices.%s", "\n");
      cytnx_error_msg(Tl.shape().size() != 1,
                      "[ERROR][Tl] tensor for Vectordot should be rank-1.%s", "\n");
      cytnx_error_msg(Tr.shape().size() != 1,
                      "[ERROR][Tr] tensor for Vectordot should be rank-1.%s", "\n");
      cytnx_error_msg(Tr.shape()[0] != Tl.shape()[0],
                      "[ERROR] two tensor for Vectordot should have same length.%s", "\n");

      Tensor L, R;
      Tensor out;
      if (Tl.dtype() != Tr.dtype()) {
        if (Tl.dtype() < Tr.dtype()) {
          L = Tl;  // this is ref!!! no copy, so please don't modify this!!
          R = Tr.astype(Tl.dtype());
          out.Init({1}, Tl.dtype(), Tl.device());
        } else {
          L = Tl.astype(Tr.dtype());
          R = Tr;  // this is ref!!! no copy, so please don't modify this!!
          out.Init({1}, Tr.dtype(), Tr.device());
        }
      } else {
        L = Tl;
        R = Tr;
        out.Init({1}, Tr.dtype(), Tr.device());
      }

      if (out.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Vd_ii[out.dtype()](
          out._impl->storage()._impl, L._impl->storage()._impl, R._impl->storage()._impl,
          L._impl->storage()._impl->size(), is_conj);
        return out;
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tl.device()));
        cytnx::linalg_internal::lii.cuVd_ii[out.dtype()](
          out._impl->storage()._impl, L._impl->storage()._impl, R._impl->storage()._impl,
          L._impl->storage()._impl->size(), is_conj);

        return out;
#else
        cytnx_error_msg(true, "[Vectordot] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return Tensor();
#endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx
