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

    void Vsplit_(std::vector<Tensor> &out, const Tensor &Tin,
                 const std::vector<cytnx_uint64> &dims) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[ERROR][Vsplit_] Can only work for rank-2 Tensor.%s", "\n");
      cytnx_error_msg(Tin.dtype() == Type.Bool,
                      "[ERROR][Vsplit_] currently does not support Bool type!%s", "\n");
      Tensor _Tn = Tin.contiguous();

      cytnx_uint64 sum = 0;
      for (auto s : dims) {
        sum += s;
        cytnx_error_msg(s == 0, "[ERROR][Vsplit_] all the elements in dims cannot be zero!%s",
                        "\n");
      }
      cytnx_error_msg(sum != Tin.shape()[0],
                      "[ERROR][Vsplit_] the sum of all elements in [dims] must match the dim of "
                      "axes [0] for Tin.%s",
                      "\n");

      out.resize(dims.size());
      std::vector<char *> targ_ptrs(dims.size());
      for (int i = 0; i < out.size(); i++) {
        out[i] = Tensor({dims[i], Tin.shape()[1]}, Tin.dtype(), Tin.device());
        targ_ptrs[i] = (char *)out[i].storage().data();
      }

      if (Tin.device() == Device.cpu) {
        algo_internal::vSplit_internal(targ_ptrs, (char *)_Tn.storage().data(), dims,
                                       _Tn.shape()[1], Type.typeSize(Tin.dtype()));
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tin.device()));
        algo_internal::cuvSplit_internal(targ_ptrs, (char *)_Tn.storage().data(), dims,
                                         _Tn.shape()[1], Type.typeSize(Tin.dtype()));
  #else
        cytnx_error_msg(
          true, "[ERROR][Vsplit_] input is on GPU but current cytnx is compiled without GPU.%s",
          "\n");
  #endif
      }
    }
    std::vector<Tensor> Vsplit(const Tensor &Tin, const std::vector<cytnx_uint64> &dims) {
      std::vector<Tensor> out;
      Vsplit_(out, Tin, dims);
      return out;
    }

  }  // namespace algo
}  // namespace cytnx

#endif
