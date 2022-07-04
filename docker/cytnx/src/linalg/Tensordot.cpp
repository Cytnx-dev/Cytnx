#include "linalg/linalg.hpp"
#include "utils/utils.hpp"

namespace cytnx {

  namespace linalg {
    Tensor Tensordot(const Tensor &Tl, const Tensor &Tr, const std::vector<cytnx_uint64> &idxl,
                     const std::vector<cytnx_uint64> &idxr) {
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

      std::vector<cytnx_uint64> mapperL, mapperR;
      std::vector<cytnx_uint64> non_contract_l =
        vec_erase(utils_internal::range_cpu(Tl.shape().size()), idxl);
      std::vector<cytnx_uint64> non_contract_r =
        vec_erase(utils_internal::range_cpu(Tr.shape().size()), idxr);

      // calculate permute
      vec_concatenate_(mapperL, non_contract_l, idxl);
      vec_concatenate_(mapperR, idxr, non_contract_r);

      // checking + calculate comm_dim:
      cytnx_int64 comm_dim = 1;
      for (cytnx_uint64 i = 0; i < idxl.size(); i++) {
        cytnx_error_msg(Tl.shape()[idxl[i]] != Tr.shape()[idxr[i]],
                        "the index L=%d and R=%d have different dimension!\n", idxl[i], idxr[i]);
        comm_dim *= Tl.shape()[idxl[i]];
      }

      // calculate output shape:
      std::vector<cytnx_int64> new_shape(non_contract_l.size() + non_contract_r.size());
      for (cytnx_uint64 i = 0; i < non_contract_l.size(); i++)
        new_shape[i] = Tl.shape()[non_contract_l[i]];
      for (cytnx_uint64 i = 0; i < non_contract_r.size(); i++)
        new_shape[non_contract_l.size() + i] = Tr.shape()[non_contract_r[i]];

      // permute!
      Tensor tmpL = Tl.permute(mapperL);
      Tensor tmpR = Tr.permute(mapperR);
      tmpL.reshape_({-1, comm_dim});
      tmpR.reshape_({comm_dim, -1});

      Tensor out = Matmul(tmpL, tmpR);
      out.reshape_(new_shape);

      return out;
    }

  }  // namespace linalg
}  // namespace cytnx
