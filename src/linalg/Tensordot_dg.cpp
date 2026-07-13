#include "linalg.hpp"
#include "utils/utils.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else

namespace cytnx {

  namespace linalg {
    namespace {
      void check_tensordot_dg_axis_bounds(const char *side,
                                          const std::vector<cytnx_uint64> &indices,
                                          const cytnx_uint64 rank) {
        for (cytnx_uint64 i = 0; i < indices.size(); i++) {
          cytnx_error_msg(indices[i] >= rank,
                          "[ERROR][Tensordot_dg] axis %s=%llu is out of bounds for rank %llu.%s",
                          side, static_cast<unsigned long long>(indices[i]),
                          static_cast<unsigned long long>(rank), "\n");
          for (cytnx_uint64 j = i + 1; j < indices.size(); j++) {
            cytnx_error_msg(indices[i] == indices[j],
                            "[ERROR][Tensordot_dg] duplicate contracted axis %s=%llu.%s", side,
                            static_cast<unsigned long long>(indices[i]), "\n");
          }
        }
      }
    }  // namespace

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
        check_tensordot_dg_axis_bounds("L", idxl, 2);
        check_tensordot_dg_axis_bounds("R", idxr, Tr.rank());
        if (idxl.size() != 1) {
          // this is weighted trace, juse expand diag into full dense and then call Tensordot.
          return Tensordot(Diag(Tl), Tr, {0, 1}, idxr);
        }
        non_contract_l = vec_erase(vec_range(2), idxl);
        non_contract_r = vec_erase(vec_range(Tr.shape().size()), idxr);
        Tlshape.push_back(Tl.shape()[0]);
        Tlshape.push_back(Tl.shape()[0]);
        Trshape = Tr.shape();

      } else {
        cytnx_error_msg(Tr.shape().size() != 1,
                        "[ERROR] diag_L=false requires Tr to be rank-1 tensor.%s", "\n");
        check_tensordot_dg_axis_bounds("L", idxl, Tl.rank());
        check_tensordot_dg_axis_bounds("R", idxr, 2);
        if (idxr.size() != 1) {
          // this is weighted trace, juse expand diag into full dense and then call Tensordot.
          return Tensordot(Tl, Diag(Tr), idxl, {0, 1});
        }

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
      cytnx_int64 left_dim = 1;
      cytnx_int64 right_dim = 1;
      for (cytnx_uint64 i = 0; i < non_contract_l.size(); i++) {
        new_shape[i] = Tlshape[non_contract_l[i]];
        left_dim *= new_shape[i];
      }
      for (cytnx_uint64 i = 0; i < non_contract_r.size(); i++) {
        new_shape[non_contract_l.size() + i] = Trshape[non_contract_r[i]];
        right_dim *= new_shape[non_contract_l.size() + i];
      }

      // permute!
      Tensor tmpL, tmpR, out, tmpout;

      if (diag_L) {
        // Both bonds of Diag will be contracted.
        if (idxl.size() == 2) {
          tmpL = Tl;
          const cytnx_int64 diag_dim = Tlshape[idxl[1]];
          tmpR = Tr.permute(mapperR).reshape({diag_dim, diag_dim * right_dim});
          tmpout = Matmul_dg(tmpL, tmpR);
          tmpout.reshape_({static_cast<cytnx_int64>(Tlshape[idxl[0]]),
                           static_cast<cytnx_int64>(Tlshape[idxl[1]]), right_dim});
          out = Trace(tmpout, 0, 1);
        } else {
          tmpL = Tl;
          tmpR = Tr.permute(mapperR).reshape({comm_dim, right_dim});
          out = Matmul_dg(tmpL, tmpR);
        }
      } else {
        if (idxr.size() == 2) {
          tmpL = Tl.permute(mapperL).reshape({left_dim, comm_dim});
          tmpR = Tr;
          tmpout = Matmul_dg(tmpL, tmpR);
          tmpout.reshape_({left_dim, static_cast<cytnx_int64>(Tlshape[idxl[1]]),
                           static_cast<cytnx_int64>(Tlshape[idxl[0]])});
          out = Trace(tmpout, 1, 2);
        } else {
          tmpL = Tl.permute(mapperL).reshape({left_dim, comm_dim});
          tmpR = Tr;
          out = Matmul_dg(tmpL, tmpR);
        }
      }

      out.reshape_(new_shape);

      return out;
    }

  }  // namespace linalg
}  // namespace cytnx

#endif
