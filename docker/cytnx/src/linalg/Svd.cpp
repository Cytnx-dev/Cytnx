#include "linalg/linalg.hpp"
#include <iostream>

namespace cytnx {
  namespace linalg {
    std::vector<Tensor> Svd(const Tensor &Tin, const bool &is_U, const bool &is_vT) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[Svd] error, Svd can only operate on rank-2 Tensor.%s", "\n");
      cytnx_error_msg(
        !Tin.is_contiguous(),
        "[Svd] error tensor must be contiguous. Call Contiguous_() or Contiguous() first%s", "\n");

      cytnx_uint64 n_singlu = std::max(cytnx_uint64(1), std::min(Tin.shape()[0], Tin.shape()[1]));

      Tensor in;
      if (Tin.dtype() > Type.Float)
        in = Tin.astype(Type.Float);
      else
        in = Tin;

      // std::cout << n_singlu << std::endl;

      Tensor U, S, vT;
      S.Init({n_singlu}, in.dtype() <= 2 ? in.dtype() + 2 : in.dtype(),
             in.device());  // if type is complex, S should be real
      if (is_U) {
        U.Init({in.shape()[0], n_singlu}, in.dtype(), in.device());
      }
      if (is_vT) {
        vT.Init({n_singlu, in.shape()[1]}, in.dtype(), in.device());
      }

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Svd_ii[in.dtype()](
          in._impl->storage()._impl, U._impl->storage()._impl, vT._impl->storage()._impl,
          S._impl->storage()._impl, in.shape()[0], in.shape()[1]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_U) out.push_back(U);
        if (is_vT) out.push_back(vT);

        return out;

      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(in.device()));
        cytnx::linalg_internal::lii.cuSvd_ii[in.dtype()](
          in._impl->storage()._impl, U._impl->storage()._impl, vT._impl->storage()._impl,
          S._impl->storage()._impl, in.shape()[0], in.shape()[1]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_U) out.push_back(U);
        if (is_vT) out.push_back(vT);

        return out;
#else
        cytnx_error_msg(true, "[Svd] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
#endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx
