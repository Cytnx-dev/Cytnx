#include "algo.hpp"
#include "Accessor.hpp"
#include "Generator.hpp"

#ifdef BACKEND_TORCH

#else

  #include "backend/algo_internal_interface.hpp"
  #include "backend/Storage.hpp"
  #include "backend/Scalar.hpp"
namespace cytnx {
  namespace algo {
    typedef Accessor ac;

    void Hsplit_(std::vector<Tensor> &out, const Tensor &Tin,
                 const std::vector<cytnx_uint64> &dims) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[ERROR][Hsplit_] Can only work for rank-2 Tensor.%s", "\n");
      cytnx_error_msg(Tin.dtype() == Type.Bool,
                      "[ERROR][Hsplit_] currently does not support Bool type!%s", "\n");
      Tensor _Tn = Tin.contiguous();

      cytnx_uint64 sum = 0;
      for (auto s : dims) {
        sum += s;
        cytnx_error_msg(s == 0, "[ERROR][Vsplit_] all the elements in dims cannot be zero!%s",
                        "\n");
      }
      cytnx_error_msg(sum != Tin.shape()[1],
                      "[ERROR][Hsplit_] the sum of all elements in [dims] must match the dim of "
                      "axes [1] for Tin.%s",
                      "\n");

      out.resize(dims.size());
      std::vector<char *> targ_ptrs(dims.size());
      for (int i = 0; i < out.size(); i++) {
        out[i] = Tensor({Tin.shape()[0], dims[i]}, Tin.dtype(), Tin.device());
        targ_ptrs[i] = (char *)out[i].storage().data();
      }

      if (Tin.device() == Device.cpu) {
        algo_internal::hSplit_internal(targ_ptrs, (char *)_Tn.storage().data(), dims,
                                       _Tn.shape()[0], Type.typeSize(Tin.dtype()));
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tin.device()));
        algo_internal::cuhSplit_internal(targ_ptrs, (char *)_Tn.storage().data(), dims,
                                         _Tn.shape()[0], Type.typeSize(Tin.dtype()));
  #else
        cytnx_error_msg(
          true, "[ERROR][Hsplit_] input is on GPU but current cytnx is compiled without GPU.%s",
          "\n");
  #endif
      }
    }

    std::vector<Tensor> Hsplit(const Tensor &Tin, const std::vector<cytnx_uint64> &dims) {
      std::vector<Tensor> out;
      Hsplit_(out, Tin, dims);
      return out;
    }

  }  // namespace algo
}  // namespace cytnx

#endif  // BACKEND_TORCH
