#include "linalg.hpp"
#include "Generator.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
#include "algo.hpp"
#include <iostream>
#include <type_traits>
#include <vector>

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {

    template <typename T>
    Tensor ExpM(const Tensor &Tin, const T &a, const T &b) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[ExpM] error, ExpM can only operate on rank-2 Tensor.%s", "\n");
      // cytnx_error_msg(!Tin.is_contiguous(), "[ExpM] error tensor must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1],
                      "[ExpM] error, ExpM can only operator on square Tensor (#row = #col%s", "\n");

      std::vector<Tensor> su = cytnx::linalg::Eig(Tin, true);
      Tensor s, u, ut;
      // exp(a*M + b*I) with a == 0 is exp(b)*I: keep the bias b (matches ExpH).
      if (a == 0) {
        if (b == 0)
          return cytnx::identity(Tin.shape()[0], Tin.dtype(), Tin.device());
        else
          return cytnx::identity(Tin.shape()[0], Tin.dtype(), Tin.device()) * exp(b);
      }

      if (b == 0)
        s = cytnx::linalg::Exp(a * su[0]);
      else
        s = cytnx::linalg::Exp(a * su[0] + b);

      u = su[1];

      //[Optim required]
      s = cytnx::linalg::Diag(s);
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
    static void ExpM_Dense_UT_internal(UniTensor &out, const UniTensor &Tin, const T &a,
                                       const T &b) {
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

    // Block-wise matrix exponential of a general symmetric UniTensor. Handles both BlockUniTensor
    // (bosonic) and BlockFermionicUniTensor (fermionic), selected by the template parameter BUT.
    // For the fermionic case, sign-flipped blocks are negated to the physical operator before each
    // per-qcharge dense ExpM, and the result is stored with an all-false signflip (it is already
    // physical). For the bosonic case there are no sign flips and those steps do nothing.
    template <class BUT, typename T>
    static void ExpM_BlockUT_internal(UniTensor &out, const UniTensor &Tin, const T &a,
                                      const T &b) {
      std::vector<bool> signflip;
      if constexpr (std::is_same_v<BUT, BlockFermionicUniTensor>)
        signflip = static_cast<BlockFermionicUniTensor *>(Tin._impl.get())->_signflip;

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
      const auto left_qnums =
        (BdLeft.type() == bondType::BD_IN) ? BdLeft.qnums() : BdLeft.calc_reverse_qnums();
      for (cytnx_uint64 b = 0; b < Tin.Nblocks(); b++) {
        mgrp[left_qnums[new_itoi[b][0]]].push_back(b);
      }

      // 4) for each qcharge in key, combining the blocks into a big chunk!
      vec2d<cytnx_uint64> &ref_itoi = ((BUT *)out._impl.get())->_inner_to_outer_idx;
      std::vector<Tensor> &out_blocks_ = ((BUT *)out._impl.get())->_blocks;
      for (auto const &x : mgrp) {
        vec2d<cytnx_uint64> itoi_indicators(x.second.size());
        for (int i = 0; i < x.second.size(); i++) {
          itoi_indicators[i] = new_itoi[x.second[i]];
        }
        auto order = vec_sort(itoi_indicators, true);
        std::vector<Tensor> Tlist(itoi_indicators.size());
        cytnx_uint64 Rblk_dim = 0;
        cytnx_int64 tmp = -1;
        cytnx_int64 row_szs;
        std::vector<cytnx_uint64> rdims, cdims;  // this is used to split!
        vec2d<cytnx_uint64> old_shape(order.size());

        for (int i = 0; i < order.size(); i++) {
          cytnx_int64 current_block = x.second[order[i]];
          Tlist[i] = Tin.get_blocks()[current_block];
          row_szs = 1;
          old_shape[i] = Tlist[i].shape();
          for (int j = 0; j < Tin.rowrank(); j++) {
            row_szs *= Tlist[i].shape()[j];
          }
          bool flip = false;
          if constexpr (std::is_same_v<BUT, BlockFermionicUniTensor>)
            flip = signflip[current_block];
          if (flip) Tlist[i] = -Tlist[i];  // negate to the physical operator before exp
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

      // the sign was already included when creating the per-qcharge blocks, so the resulting
      // signflip is all false.
      if constexpr (std::is_same_v<BUT, BlockFermionicUniTensor>)
        ((BUT *)out._impl.get())->_signflip = std::vector<bool>(out_blocks_.size(), false);
    }

    template <typename T>
    UniTensor ExpM(const UniTensor &Tin, const T &a, const T &b) {
      cytnx_error_msg(Tin.rowrank() < 1,
                      "[ERROR][ExpM] Input UniTensor should have rowrank > 0, but rowrank is %d\n",
                      Tin.rowrank());
      cytnx_error_msg(Tin.rowrank() >= Tin.rank(),
                      "[ERROR][ExpM] Input UniTensor should have rowrank < rank, but rowrank is %d "
                      "and rank is %d\n",
                      Tin.rowrank(), Tin.rank());

      if (Tin.uten_type() == UTenType.Dense) {
        UniTensor out;
        if (Tin.is_contiguous()) {
          out = Tin.clone();
        } else {
          out = Tin.contiguous();
        }
        ExpM_Dense_UT_internal(out, Tin, a, b);
        return out;
      } else if (Tin.uten_type() == UTenType.Block) {
        // copy everything except _blocks and _inner_to_outer_idx
        BlockUniTensor *raw_out = ((BlockUniTensor *)Tin._impl.get())->clone_meta(false, true);
        UniTensor out;
        out._impl = boost::intrusive_ptr<UniTensor_base>(raw_out);
        ExpM_BlockUT_internal<BlockUniTensor>(out, Tin, a, b);
        return out;
      } else if (Tin.uten_type() == UTenType.BlockFermionic) {
        // copy everything except _blocks and _inner_to_outer_idx
        BlockFermionicUniTensor *raw_out =
          ((BlockFermionicUniTensor *)Tin._impl.get())->clone_meta(false, true);
        UniTensor out;
        out._impl = boost::intrusive_ptr<UniTensor_base>(raw_out);
        ExpM_BlockUT_internal<BlockFermionicUniTensor>(out, Tin, a, b);
        return out;
      } else {
        cytnx_error_msg(Tin.uten_type() == UTenType.Void,
                        "[ERROR] UniTensor is not initialized and of type Void.%s", "\n");
        cytnx_error_msg(
          Tin.uten_type() == UTenType.Sparse,
          "[ERROR] SparseUniTensor is deprecated. Use BlockUniTensor or LinOp instead.%s", "\n");
        cytnx_error_msg(true, "[ERROR][ExpM] UniTensor type '%s' not supported\n",
                        Tin.uten_type_str().c_str());
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
