#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include <iostream>
#include "Tensor.hpp"

namespace cytnx {
  namespace linalg {
    std::vector<Tensor> Eig(const Tensor &Tin, const bool &is_V, const bool &row_v) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[Eig] error, Eig can only operate on rank-2 Tensor.%s", "\n");
      // cytnx_error_msg(!Tin.is_contiguous(), "[Eig] error tensor must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1],
                      "[Eig] error, Eig should accept a square matrix%s", "\n");

      Tensor in = Tin.contiguous();
      if (Tin.dtype() > Type.Float)
        in = in.astype(Type.ComplexDouble);
      else {
        if (Tin.dtype() == Type.Float)
          in = in.astype(Type.ComplexFloat);
        else if (Tin.dtype() == Type.Double)
          in = in.astype(Type.ComplexDouble);
      }

      Tensor S, V;
      S.Init({in.shape()[0]}, in.dtype() % 2 == 1 ? Type.ComplexDouble : Type.ComplexFloat,
             in.device());  // S should always be double!!

      if (is_V) {
        V.Init(in.shape(), in.dtype(), in.device()); /*V.storage().set_zeros();*/
      }

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Eig_ii[in.dtype()](in._impl->storage()._impl,
                                                       S._impl->storage()._impl,
                                                       V._impl->storage()._impl, in.shape()[0]);

        std::vector<Tensor> out;
        out.push_back(S);
        // std::cout << "[Eig][V]" << std::endl;
        // std::cout << V << std::endl;
        if (is_V) {
          out.push_back(V);
          if (!row_v) {
            if (out.back().dtype() == Type.ComplexFloat ||
                out.back().dtype() == Type.ComplexDouble) {
              out.back().permute_({1, 0}).contiguous_();
              // std::cout << out.back();
              out.back().Conj_();
              // std::cout << out.back();
              // std::cout << "ok" << std::endl;
            } else
              out.back().permute_({1, 0}).contiguous_();
          }
        }

        return out;

      } else {
#ifdef UNI_GPU
        /*
        checkCudaErrors(cudaSetDevice(in.device()));
        cytnx::linalg_internal::lii.cuEig_ii[in.dtype()](in._impl->storage()._impl,
                                                S._impl->storage()._impl,
                                                V._impl->storage()._impl,
                                                in.shape()[0]);

        std::vector<Tensor> out;
        out.push_back(S);
        if(is_V){
            out.push_back(V);
            if(!row_v) out.back().permute_({1,0}).contiguous_();
        }
        */
        cytnx_error_msg(true, "[ERROR]currently Eig for non-symmetric matrix is not supported.%s",
                        "\n");
        return std::vector<Tensor>();
#else
        cytnx_error_msg(true, "[Eig] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return std::vector<Tensor>();
#endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx
