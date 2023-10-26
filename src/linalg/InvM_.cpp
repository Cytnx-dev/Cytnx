#include "linalg.hpp"

#include "Tensor.hpp"
using namespace std;
#ifdef BACKEND_TORCH
#else
  #include "../backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {
    void InvM_(Tensor &Tin) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[InvM] error, InvM can only operate on rank-2 Tensor.%s", "\n");
      // cytnx_error_msg(!Tin.is_contiguous(), "[InvM] error tensor must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1],
                      "[InvM] error, the size of last two rank should be the same.%s", "\n");

      if (Tin.dtype() > 4) Tin = Tin.contiguous().astype(Type.Double);

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.InvM_inplace_ii[Tin.dtype()](Tin._impl->storage()._impl,
                                                                 Tin.shape().back());

      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tin.device()));
        cytnx::linalg_internal::lii.cuInvM_inplace_ii[Tin.dtype()](Tin._impl->storage()._impl,
                                                                   Tin.shape().back());

  #else
        cytnx_error_msg(true, "[InvM] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
  #endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx

namespace cytnx {
  namespace linalg {
    void _InvM_inplace_Dense_UT(UniTensor &Tin) {
      Tensor tmp;

      if (Tin.is_contiguous())
        tmp = Tin.get_block_();
      else {
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

      cytnx::linalg::InvM_(tmp);

      if (Tin.is_contiguous()) tmp.reshape_(oldshape);
    }
    void InvM_(UniTensor &Tin) {
      cytnx_error_msg(Tin.rowrank() < 1 || Tin.rank() == 1,
                      "[InvM_][ERROR] InvM_ for UniTensor should have rank>1 and rowrank>0%s",
                      "\n");

      cytnx_error_msg(Tin.is_diag(),
                      "[InvM_[ERROR] InvM_ for diagonal UniTensor is trivial and currently not "
                      "support. Use other manipulation.%s",
                      "\n");
      if (Tin.uten_type() == UTenType.Dense) {
        _InvM_inplace_Dense_UT(Tin);

      } else {
        cytnx_error_msg(true,
                        "[ERROR] InvM_, unsupported type of UniTensor only support (Dense). "
                        "something wrong internal%s",
                        "\n");
      }
    }

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
