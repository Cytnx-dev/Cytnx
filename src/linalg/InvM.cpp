#include "linalg.hpp"

#include "Tensor.hpp"
using namespace std;

#ifdef BACKEND_TORCH
#else
  #include "../backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {
    Tensor InvM(const Tensor &Tin) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[InvM] error, InvM can only operate on rank-2 Tensor.%s", "\n");
      // cytnx_error_msg(!Tin.is_contiguous(), "[InvM] error tensor must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1],
                      "[InvM] error, the size of last two rank should be the same.%s", "\n");

      Tensor out;
      if (!Tin.is_contiguous())
        out = Tin.contiguous();
      else
        out = Tin.clone();

      if (Tin.dtype() > 4) out = out.astype(Type.Double);

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.InvM_inplace_ii[out.dtype()](out._impl->storage()._impl,
                                                                 out.shape().back());

        return out;

      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(out.device()));
        cytnx::linalg_internal::lii.cuInvM_inplace_ii[out.dtype()](out._impl->storage()._impl,
                                                                   out.shape().back());
        return out;
  #else
        cytnx_error_msg(true, "[InvM] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return Tensor();
  #endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx

namespace cytnx {
  namespace linalg {
    void _InvM_Dense_UT(UniTensor &outCyT, const UniTensor &Tin) {
      Tensor tmp;

      if (Tin.is_contiguous()) {
        tmp = Tin.get_block_();
      } else {
        tmp = Tin.get_block();
        tmp.contiguous_();
      }

      vector<cytnx_uint64> tmps = tmp.shape();
      vector<cytnx_int64> oldshape(tmps.begin(), tmps.end());
      tmps.clear();
      vector<string> oldlabel = Tin.labels();

      // collapse as Matrix:
      cytnx_int64 rowdim = 1;
      for (cytnx_uint64 i = 0; i < Tin.rowrank(); i++) rowdim *= tmp.shape()[i];
      tmp.reshape_({rowdim, -1});

      Tensor outT = cytnx::linalg::InvM(tmp);

      if (Tin.is_contiguous()) tmp.reshape_(oldshape);

      outCyT = UniTensor(outT, false, 1);
    }

    UniTensor InvM(const UniTensor &Tin) {
      cytnx_error_msg(Tin.rowrank() < 1 || Tin.rank() == 1,
                      "[InvM_][ERROR] InvM for UniTensor should have rank>1 and rowrank>0%s", "\n");

      cytnx_error_msg(Tin.is_diag(),
                      "[InvM_[ERROR] InvM for diagonal UniTensor is trivial and currently not "
                      "support. Use other manipulation.%s",
                      "\n");
      UniTensor outCyT;
      if (Tin.uten_type() == UTenType.Dense) {
        _InvM_Dense_UT(outCyT, Tin);
        return outCyT;
      } else {
        cytnx_error_msg(true,
                        "[ERROR] InvM, unsupported type of UniTensor only support (Dense). "
                        "something wrong internal%s",
                        "\n");
        return UniTensor();
      }
    }

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
