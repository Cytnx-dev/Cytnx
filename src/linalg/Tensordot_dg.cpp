#include "linalg.hpp"
#include "utils/utils.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"

namespace cytnx {

  namespace linalg {
    Tensor Tensordot_dg(const Tensor &Tl, const Tensor &Tr, const std::vector<cytnx_uint64> &idxl,
                        const std::vector<cytnx_uint64> &idxr, const bool &diag_L) {
      // checking:
      cytnx_error_msg(idxl.size() != idxr.size(),
                      "[ERROR] the number of index to trace must be consist across two tensors.%s",
                      "\n");
      cytnx_error_msg(
        idxl.size() == 0,
        "[ERROR] pass empty index list for trace. suggestion: call linalg::Otimes() instead?%s",
        "\n");
      cytnx_error_msg(Tl.device() != Tr.device(),
                      "[ERROR] two tensor for Tensordot cannot on different devices.%s", "\n");

      if ((Tl.shape().size() != 1) && (Tr.shape().size() != 1))
        cytnx_error_msg(true,
                        "[ERROR] dg version of Tensordot requires one of them to be rank-1 input "
                        "as a diagonal rank-2 tensor.%s",
                        "\n");

      std::vector<cytnx_uint64> mapperL, mapperR, non_contract_l, non_contract_r;
      std::vector<cytnx_uint64> Tlshape, Trshape;
      if (diag_L) {
        cytnx_error_msg(Tl.shape().size() != 1,
                        "[ERROR] diag_L=true requires Tl to be rank-1 tensor.%s", "\n");
        non_contract_l = vec_erase(vec_range(2), idxl);
        non_contract_r = vec_erase(vec_range(Tr.shape().size()), idxr);
        Tlshape.push_back(Tl.shape()[0]);
        Tlshape.push_back(Tl.shape()[0]);
        Trshape = Tr.shape();

      } else {
        cytnx_error_msg(Tr.shape().size() != 1,
                        "[ERROR] diag_L=false requires Tr to be rank-1 tensor.%s", "\n");
        non_contract_l = vec_erase(vec_range(Tl.shape().size()), idxl);
        non_contract_r = vec_erase(vec_range(2), idxr);
        Trshape.push_back(Tr.shape()[0]);
        Trshape.push_back(Tr.shape()[0]);
        Tlshape = Tl.shape();
      }

      // calculate permute
      vec_concatenate_(mapperL, non_contract_l, idxl);
      vec_concatenate_(mapperR, idxr, non_contract_r);

      // checking + calculate comm_dim:
      cytnx_int64 comm_dim = 1;
      for (cytnx_uint64 i = 0; i < idxl.size(); i++) {
        cytnx_error_msg(Tlshape[idxl[i]] != Trshape[idxr[i]],
                        "the index L=%d and R=%d have different dimension!\n", idxl[i], idxr[i]);
        comm_dim *= Tlshape[idxl[i]];
      }

      // calculate output shape:
      std::vector<cytnx_int64> new_shape(non_contract_l.size() + non_contract_r.size());
      for (cytnx_uint64 i = 0; i < non_contract_l.size(); i++)
        new_shape[i] = Tlshape[non_contract_l[i]];
      for (cytnx_uint64 i = 0; i < non_contract_r.size(); i++)
        new_shape[non_contract_l.size() + i] = Trshape[non_contract_r[i]];

      if (new_shape.size() == 0) {
        new_shape.push_back(1);
      }

      // permute!
      Tensor tmpL, tmpR, out, tmpout;

      if (diag_L) {
        // Both bonds of Diag will be contracted.
        if (idxl.size() == 2) {
          tmpL = Tl;
          tmpR = Tr.permute(mapperR).reshape({Tlshape[idxl[1]], -1});
          tmpout = Matmul_dg(tmpL, tmpR);
          tmpout.reshape_({Tlshape[idxl[0]], Tlshape[idxl[1]], -1});
          out = Trace(tmpout, 0, 1);
        } else {
          tmpL = Tl;
          tmpR = Tr.permute(mapperR).reshape({comm_dim, -1});
          out = Matmul_dg(tmpL, tmpR);
        }
      } else {
        if (idxr.size() == 2) {
          tmpL = Tl.permute(mapperL).reshape({-1, comm_dim});
          tmpR = Tr;
          tmpout = Matmul_dg(tmpL, tmpR);
          tmpout.reshape_({-1, Tlshape[idxl[1]], Tlshape[idxl[0]]});
          out = Trace(tmpout, 1, 2);
        } else {
          tmpL = Tl.permute(mapperL).reshape({-1, comm_dim});
          tmpR = Tr;
          out = Matmul_dg(tmpL, tmpR);
        }
      }

      out.reshape_(new_shape);

      return out;
    }

  }  // namespace linalg
}  // namespace cytnx
