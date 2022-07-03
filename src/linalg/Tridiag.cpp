#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include <iostream>
#include "Tensor.hpp"

namespace cytnx {
  namespace linalg {
    std::vector<Tensor> Tridiag(const Tensor &Diag, const Tensor &Sub_diag, const bool &is_V,
                                const bool &is_row) {
      cytnx_error_msg(Diag.shape().size() != 1,
                      "[Tridiag] error, Tridiag can only accept on vector (rank-1) Tensor.%s",
                      "\n");
      // cytnx_error_msg(!Diag.is_contiguous(), "[Tridiag] error tensor [#1][Diag] must be
      // contiguous. Call Contiguous_() or Contiguous() first%s","\n");
      cytnx_error_msg(Sub_diag.shape().size() != 1,
                      "[Tridiag] error tensor [#2][SubDiag] must be vector (rank-1) Tensor%s",
                      "\n");
      cytnx_error_msg(
        Diag.device() != Sub_diag.device(),
        "[Tridiag] error, two input tensors must in the same device. Call to() or to_() first%s",
        "\n");
      cytnx_error_msg(Diag.dtype() <= 2 || Sub_diag.dtype() <= 2,
                      "[Tridiag] error, tri-diagonalize can only accept real vectors%s", "\n");

      // check prior type:
      unsigned int cType;
      if (Diag.dtype() < Sub_diag.dtype()) {
        cType = Diag.dtype();
      } else {
        cType = Sub_diag.dtype();
      }
      if (cType > Type.Float) cType = Type.Double;

      Tensor in_diag, s_diag;
      if (Diag.dtype() > cType)
        in_diag = Diag.astype(cType);
      else
        in_diag = Diag;

      if (Sub_diag.dtype() > cType)
        s_diag = Sub_diag.astype(cType);
      else
        s_diag = Sub_diag;

      // std::cout << s_diag.shape() << std::endl;
      // std::cout << in_diag << std::endl;
      // std::cout << s_diag << std::endl;
      Tensor vT, S;
      S.Init({Diag.shape()[0]}, cType <= 2 ? cType + 2 : cType,
             Diag.device());  // if type is complex, S should be real
      if (is_V) {
        // cytnx_error_msg((k<1)||(k>Diag.shape()[0]),"[Tridiag] error, number of eigen vector k
        // should be >1 and <=L%s","\n");
        vT.Init({Diag.shape()[0], Diag.shape()[0]}, cType, in_diag.device());
      }

      if (Diag.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Td_ii[cType](
          in_diag._impl->storage()._impl, s_diag._impl->storage()._impl, S._impl->storage()._impl,
          vT._impl->storage()._impl, in_diag.shape()[0]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_V) {
          out.push_back(vT);
          if (!is_row) out.back().permute_(1, 0);
        }

        return out;

      } else {
        cytnx_error_msg(true, "[ERROR]Curretly, Tridiagonal does not have GPU version.%s", "\n");
        /*
        #ifdef UNI_GPU
            checkCudaErrors(cudaSetDevice(in.device()));
            cytnx::linalg_internal::lii.cuSvd_ii[in.dtype()](in._impl->storage()._impl,
                                                    U._impl->storage()._impl,
                                                    vT._impl->storage()._impl,
                                                    S._impl->storage()._impl,in.shape()[0],in.shape()[1]);

            std::vector<Tensor> out;
            out.push_back(S);
            if(is_U) out.push_back(U);
            if(is_vT) out.push_back(vT);

            return out;
        #else
            cytnx_error_msg(true,"[Svd] fatal error,%s","try to call the gpu section without CUDA
        support.\n"); return std::vector<Tensor>(); #endif
        */
      }
    }

  }  // namespace linalg
}  // namespace cytnx
