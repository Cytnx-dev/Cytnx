#include "linalg/linalg.hpp"
#include <iostream>

namespace cytnx {
  namespace linalg {
    std::vector<Tensor> Eigh(const Tensor &Tin, const bool &is_V) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[Eigh] error, Eigh can only operate on rank-2 Tensor.%s", "\n");
      cytnx_error_msg(
        !Tin.is_contiguous(),
        "[Eigh] error tensor must be contiguous. Call Contiguous_() or Contiguous() first%s", "\n");

      cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1],
                      "[Eigh] error, Eigh should accept a Hermition Tensor%s", "\n");

      Tensor in;
      if (Tin.dtype() > Type.Float)
        in = Tin.astype(Type.Float);
      else
        in = Tin;

      Tensor S, V;
      S.Init({in.shape()[0]}, in.dtype() <= 2 ? in.dtype() + 2 : in.dtype(),
             in.device());  // if type is complex, S should be real
      if (is_V) {
        V.Init(in.shape(), in.dtype(), in.device());
      }

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Eigh_ii[in.dtype()](in._impl->storage()._impl,
                                                        S._impl->storage()._impl,
                                                        V._impl->storage()._impl, in.shape()[0]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_V) out.push_back(V);

        return out;

      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(in.device()));
        cytnx::linalg_internal::lii.cuEigh_ii[in.dtype()](in._impl->storage()._impl,
                                                          S._impl->storage()._impl,
                                                          V._impl->storage()._impl, in.shape()[0]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_V) out.push_back(V);

        return out;
#else
        cytnx_error_msg(true, "[Eigh] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
#endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx
