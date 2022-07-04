#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Generator.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
#include <iostream>
#include <vector>

using namespace std;
namespace cytnx {
  namespace linalg {

    Tensor ExpH(const Tensor &Tin, const cytnx_double &a, const cytnx_double &b) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[ExpH] error, ExpH can only operate on rank-2 Tensor.%s", "\n");
      // cytnx_error_msg(!Tin.is_contiguous(), "[ExpH] error tensor must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1],
                      "[ExpH] error, ExpH can only operator on square Tensor (#row = #col%s", "\n");

      vector<Tensor> su = cytnx::linalg::Eigh(Tin, true);
      Tensor s, u, ut;
      if (a == 0) return cytnx::linalg::Diag(cytnx::ones(Tin.shape()[0]));

      if (b == 0)
        s = cytnx::linalg::Exp(a * su[0]);
      else
        s = cytnx::linalg::Exp(a * su[0] + b);

      u = su[1];
      su.clear();

      //[Optim required]
      // cout << s << endl;
      s = cytnx::linalg::Diag(s);
      // cout << s << endl;
      // cout << u;
      ut = su[1].clone();
      if (Type.is_complex(ut.dtype())) {
        ut.permute_({1, 0}).Conj_();
      } else {
        ut.permute_({1, 0});
      }
      ut.contiguous_();
      // cout << ut;

      ut = cytnx::linalg::Matmul(s, ut);
      ut = cytnx::linalg::Matmul(u, ut);

      return ut;
    }

  }  // namespace linalg

}  // namespace cytnx

namespace cytnx {
  namespace linalg {
    UniTensor ExpH(const UniTensor &Tin, const cytnx_double &a, const cytnx_double &b) {
      if (Tin.uten_type() == UTenType.Sparse) {
        UniTensor out;
        if (Tin.is_contiguous())
          out = Tin.clone();
        else
          out = Tin.contiguous();

        std::vector<Tensor> &tmp = out.get_blocks_();

        for (int i = 0; i < tmp.size(); i++) {
          tmp[i] = cytnx::linalg::ExpH(tmp[i], a, b);
        }

        return out;

      } else {
        cytnx_error_msg(
          (Tin.rowrank() == 0) || (Tin.rowrank() == Tin.rank()),
          "[ERROR][ExpH] The total dimension of row-space and col-space should be equal!!%s", "\n");
        // cytnx_error_msg(Tin.is_contiguous()==false,"[ERROR][ExpH] UniTensor must be contiguous.
        // Call Contiguous_() or Contiguous() first%s","\n"); UniTensor out = Tin.clone();
        UniTensor out;
        if (Tin.is_contiguous()) {
          out = Tin.clone();
        } else {
          out = Tin.contiguous();
        }
        cytnx_int64 Drow = 1, Dcol = 1;
        for (int i = 0; i < Tin.rowrank(); i++) {
          Drow *= Tin.shape()[i];
        }
        for (int i = Tin.rowrank(); i < Tin.rank(); i++) {
          Dcol *= Tin.shape()[i];
        }
        cytnx_error_msg(
          Drow != Dcol,
          "[ERROR][ExpH] The total dimension of row-space and col-space should be equal!!%s", "\n");

        out.get_block_().reshape_({Drow, Dcol});

        out.get_block_() = cytnx::linalg::ExpH(out.get_block_(), a, b);

        out.get_block_().reshape_(Tin.shape());

        return out;
      }
    }
  }  // namespace linalg
}  // namespace cytnx
