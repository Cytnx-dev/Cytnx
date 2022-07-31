#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include <iostream>
#include "Tensor.hpp"

namespace cytnx {
  namespace linalg {
    std::vector<Tensor> Eigh(const Tensor &Tin, const bool &is_V, const bool &row_v) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[Eigh] error, Eigh can only operate on rank-2 Tensor.%s", "\n");
      // cytnx_error_msg(!Tin.is_contiguous(), "[Eigh] error tensor must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1],
                      "[Eigh] error, Eigh should accept a Hermition Tensor%s", "\n");

      // std::cout << Tin << std::endl;
      Tensor in = Tin.contiguous();
      if (Tin.dtype() > Type.Float) in = in.astype(Type.Double);

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
        if (is_V) {
          out.push_back(V);
          if (!row_v) {
            if (out.back().dtype() == Type.ComplexFloat ||
                out.back().dtype() == Type.ComplexDouble) {
              out.back().permute_({1, 0}).contiguous_();
              out.back().Conj_();
              // std::cout << "ok";
            } else
              out.back().permute_({1, 0}).contiguous_();
          }
        }

        return out;

      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(in.device()));
        cytnx::linalg_internal::lii.cuEigh_ii[in.dtype()](in._impl->storage()._impl,
                                                          S._impl->storage()._impl,
                                                          V._impl->storage()._impl, in.shape()[0]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_V) {
          out.push_back(V);
          if (!row_v) {
            if (out.back().dtype() == Type.ComplexFloat ||
                out.back().dtype() == Type.ComplexDouble) {
              out.back().permute_({1, 0}).contiguous_();
              out.back().Conj_();
              // std::cout << "ok";
            } else
              out.back().permute_({1, 0}).contiguous_();
          }
        }

        return out;
#else
        cytnx_error_msg(true, "[Eigh] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return std::vector<Tensor>();
#endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx
