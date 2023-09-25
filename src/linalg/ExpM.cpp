#include "linalg.hpp"
#include "Generator.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
#include "algo.hpp"
#include <iostream>
#include <vector>

#ifdef BACKEND_TORCH
#else
  #include "../backend/linalg_internal_interface.hpp"

using namespace std;
namespace cytnx {
  namespace linalg {

    template <typename T>
    Tensor ExpM(const Tensor &Tin, const T &a, const T &b) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[ExpH] error, ExpH can only operate on rank-2 Tensor.%s", "\n");
      // cytnx_error_msg(!Tin.is_contiguous(), "[ExpH] error tensor must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1],
                      "[ExpH] error, ExpM can only operator on square Tensor (#row = #col%s", "\n");

      vector<Tensor> su = cytnx::linalg::Eig(Tin, true);
      // cout << su[0] << su[1] << endl;
      Tensor s, u, ut;
      if (a == 0) return cytnx::linalg::Diag(cytnx::ones(Tin.shape()[0]));

      if (b == 0)
        s = cytnx::linalg::Exp(a * su[0]);
      else
        s = cytnx::linalg::Exp(a * su[0] + b);

      u = su[1];

      //[Optim required]
      // cout << s << endl;
      s = cytnx::linalg::Diag(s);
      // cout << s << endl;
      // cout << u;
      ut = InvM(su[1]);

      ut = cytnx::linalg::Matmul(s, ut);
      ut = cytnx::linalg::Matmul(u, ut);

      return ut;
    }
    Tensor ExpM(const Tensor &Tin) { return linalg::ExpM(Tin, double(1), double(0)); }
    template Tensor ExpM(const Tensor &Tin, const cytnx_complex128 &a, const cytnx_complex128 &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_complex64 &a, const cytnx_complex64 &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_double &a, const cytnx_double &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_float &a, const cytnx_float &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_uint64 &a, const cytnx_uint64 &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_uint32 &a, const cytnx_uint32 &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_uint16 &a, const cytnx_uint16 &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_int64 &a, const cytnx_int64 &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_int32 &a, const cytnx_int32 &b);
    template Tensor ExpM(const Tensor &Tin, const cytnx_int16 &a, const cytnx_int16 &b);

  }  // namespace linalg

}  // namespace cytnx

namespace cytnx {
  namespace linalg {

    template <typename T>
    void _expm_Dense_UT(UniTensor &out, const UniTensor &Tin, const T &a, const T &b) {
      cytnx_int64 Drow = 1, Dcol = 1;
      for (int i = 0; i < Tin.rowrank(); i++) {
        Drow *= Tin.shape()[i];
      }
      for (int i = Tin.rowrank(); i < Tin.rank(); i++) {
        Dcol *= Tin.shape()[i];
      }
      cytnx_error_msg(
        Drow != Dcol,
        "[ERROR][ExpM] The total dimension of row-space and col-space should be equal!!%s", "\n");

      out.get_block_().reshape_({Drow, Dcol});

      out.get_block_() = cytnx::linalg::ExpM(out.get_block_(), a, b);

      out.get_block_().reshape_(Tin.shape());
    }

    template <typename T>
    void _expm_Sparse_UT(UniTensor &out, const UniTensor &Tin, const T &a, const T &b) {
      std::vector<Tensor> &tmp = out.get_blocks_();

      for (int i = 0; i < tmp.size(); i++) {
        tmp[i] = cytnx::linalg::ExpM(tmp[i], a, b);
      }
    }

    template <typename T>
    void _expm_Block_UT(UniTensor &out, const UniTensor &Tin, const T &a, const T &b) {
      // 1) getting the combineBond L and combineBond R for qnum list without grouping:
      //
      //   BDLeft -[ ]- BDRight
      //
      std::vector<cytnx_uint64> strides;
      strides.reserve(Tin.rank());
      auto BdLeft = Tin.bonds()[0].clone();
      for (int i = 1; i < Tin.rowrank(); i++) {
        strides.push_back(Tin.bonds()[i].qnums().size());
        BdLeft._impl->force_combineBond_(Tin.bonds()[i]._impl, false);  // no grouping
      }
      // std::cout << BdLeft << std::endl;
      strides.push_back(1);
      auto BdRight = Tin.bonds()[Tin.rowrank()].clone();
      for (int i = Tin.rowrank() + 1; i < Tin.rank(); i++) {
        strides.push_back(Tin.bonds()[i].qnums().size());
        BdRight._impl->force_combineBond_(Tin.bonds()[i]._impl, false);  // no grouping
      }
      strides.push_back(1);

      // 2) making new inner_to_outer_idx lists for each block:
      // -> a. get stride:
      for (int i = Tin.rowrank() - 2; i >= 0; i--) {
        strides[i] *= strides[i + 1];
      }
      for (int i = Tin.rank() - 2; i >= Tin.rowrank(); i--) {
        strides[i] *= strides[i + 1];
      }
      // std::cout << strides << std::endl;
      //  ->b. calc new inner_to_outer_idx!
      vec2d<cytnx_uint64> new_itoi(Tin.Nblocks(), std::vector<cytnx_uint64>(2));

      int cnt;
      for (cytnx_uint64 b = 0; b < Tin.Nblocks(); b++) {
        const std::vector<cytnx_uint64> &tmpv = Tin.get_qindices(b);
        for (cnt = 0; cnt < Tin.rowrank(); cnt++) {
          new_itoi[b][0] += tmpv[cnt] * strides[cnt];
        }
        for (cnt = Tin.rowrank(); cnt < Tin.rank(); cnt++) {
          new_itoi[b][1] += tmpv[cnt] * strides[cnt];
        }
      }

      // 3) categorize:
      // key = qnum, val = list of block locations:
      std::map<std::vector<cytnx_int64>, std::vector<cytnx_int64>> mgrp;
      for (cytnx_uint64 b = 0; b < Tin.Nblocks(); b++) {
        mgrp[BdLeft.qnums()[new_itoi[b][0]]].push_back(b);
      }

      // 4) for each qcharge in key, combining the blocks into a big chunk!
      vec2d<cytnx_uint64> &ref_itoi = ((BlockUniTensor *)out._impl.get())->_inner_to_outer_idx;
      std::vector<Tensor> &out_blocks_ = ((BlockUniTensor *)out._impl.get())->_blocks;
      for (auto const &x : mgrp) {
        vec2d<cytnx_uint64> itoi_indicators(x.second.size());
        // cout << x.second.size() << "-------" << endl;
        for (int i = 0; i < x.second.size(); i++) {
          itoi_indicators[i] = new_itoi[x.second[i]];
          // std::cout << new_itoi[x.second[i]] << std::endl;
        }
        auto order = vec_sort(itoi_indicators, true);
        std::vector<Tensor> Tlist(itoi_indicators.size());
        cytnx_uint64 Rblk_dim = 0;
        cytnx_int64 tmp = -1;
        cytnx_int64 row_szs;
        std::vector<cytnx_uint64> rdims, cdims;  // this is used to split!
        vec2d<cytnx_uint64> old_shape(order.size());

        for (int i = 0; i < order.size(); i++) {
          Tlist[i] = Tin.get_blocks()[x.second[order[i]]];
          row_szs = 1;
          old_shape[i] = Tlist[i].shape();
          for (int j = 0; j < Tin.rowrank(); j++) {
            row_szs *= Tlist[i].shape()[j];
          }
          Tlist[i] = Tlist[i].reshape({row_szs, -1});
          if (itoi_indicators[i][0] != tmp) {
            tmp = itoi_indicators[i][0];
            Rblk_dim++;
            rdims.push_back(row_szs);
          }
        }
        cytnx_error_msg(Tlist.size() % Rblk_dim, "[Internal ERROR] Tlist is not complete!%s", "\n");
        cytnx_uint64 Cblk_dim = Tlist.size() / Rblk_dim;
        for (int i = 0; i < Cblk_dim; i++) {
          cdims.push_back(Tlist[i].shape()[1]);
        }
        // BTen is the big block!!
        Tensor BTen = algo::_fx_Matric_combine(Tlist, Rblk_dim, Cblk_dim);

        BTen = cytnx::linalg::ExpM(BTen, a, b);

        Tlist.clear();
        algo::_fx_Matric_split(Tlist, BTen, rdims, cdims);

        // resize:
        for (int i = 0; i < Tlist.size(); i++) {
          Tlist[i].reshape_(old_shape[i]);
        }

        // put into new blocks:
        out_blocks_.insert(out_blocks_.end(), Tlist.begin(), Tlist.end());

        // rebuild itoi:
        for (int i = 0; i < order.size(); i++) {
          ref_itoi.push_back(Tin.get_qindices(x.second[order[i]]));
        }

      }  // for each qcharge
    }

    template <typename T>
    UniTensor ExpM(const UniTensor &Tin, const T &a, const T &b) {
      if (Tin.uten_type() == UTenType.Dense) {
        cytnx_error_msg((Tin.rowrank() == 0) || (Tin.rowrank() == Tin.rank()),
                        "[ERROR][ExpM] Rowrank must be >0 and <Tin.rank() !!%s", "\n");
        UniTensor out;
        if (Tin.is_contiguous()) {
          out = Tin.clone();
        } else {
          out = Tin.contiguous();
        }
        _expm_Dense_UT(out, Tin, a, b);

        return out;
      } else if (Tin.uten_type() == UTenType.Block) {
        cytnx_error_msg((Tin.rowrank() == 0) || (Tin.rowrank() == Tin.rank()),
                        "[ERROR][ExpM] Rowrank must be >0 and <Tin.rank() !!%s", "\n");

        // copy everything except _blocks and _inner_to_outer_idx
        BlockUniTensor *raw_out = ((BlockUniTensor *)Tin._impl.get())->clone_meta(false, true);

        UniTensor out;
        out._impl = boost::intrusive_ptr<UniTensor_base>(raw_out);

        _expm_Block_UT(out, Tin, a, b);

        return out;

      } else {
        // cytnx_error_msg(Tin.is_contiguous()==false, "[ERROR][ExpM] currently ExpM on symmetric
        // UniTensor have to operate on contiguous(). Call contiguous_() or contiguous()
        // first,%s","\n");

        UniTensor out;
        if (Tin.is_contiguous())
          out = Tin.clone();
        else
          out = Tin.contiguous();

        _expm_Sparse_UT(out, Tin, a, b);
        return out;
      }
    }

    UniTensor ExpM(const UniTensor &Tin) { return linalg::ExpM(Tin, double(1), double(0)); }

    template UniTensor ExpM(const UniTensor &Tin, const cytnx_complex128 &a,
                            const cytnx_complex128 &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_complex64 &a,
                            const cytnx_complex64 &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_double &a, const cytnx_double &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_float &a, const cytnx_float &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_uint16 &a, const cytnx_uint16 &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_uint32 &a, const cytnx_uint32 &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_uint64 &a, const cytnx_uint64 &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_int16 &a, const cytnx_int16 &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_int32 &a, const cytnx_int32 &b);
    template UniTensor ExpM(const UniTensor &Tin, const cytnx_int64 &a, const cytnx_int64 &b);

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
