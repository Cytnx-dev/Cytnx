#include "linalg.hpp"

#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "Tensor.hpp"
#include "UniTensor.hpp"
#include "algo.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {

    std::vector<Tensor> Eigh(const Tensor &Tin, const bool &is_V, const bool &row_v) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[Eigh] error, Eigh can only operate on rank-2 Tensor.%s", "\n");

      cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1],
                      "[Eigh] error, Eigh should accept a square matrix.%s", "\n");

      Tensor in = Tin.contiguous();
      if (Tin.dtype() > Type.Float) in = in.astype(Type.Double);

      Tensor S, V;
      S.Init({in.shape()[0]}, Type.to_real(in.dtype()),
             in.device());  // if type is complex, S should be real
      // V is only allocated when eigenvectors are requested. When is_V == false, V stays an empty
      // (Void) tensor; it is still passed to the backend below, which detects the Void storage and
      // calls LAPACK with jobs='N' (eigenvectors not computed), so the empty V is never written to.
      if (is_V) {
        V.Init(in.shape(), in.dtype(), in.device());
      }

      if (in.is_empty()) {
        std::vector<Tensor> out{S};
        if (is_V) out.push_back(V);
        return out;
      }

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Eigh_ii[in.dtype()](in._impl->storage()._impl,
                                                        S._impl->storage()._impl,
                                                        V._impl->storage()._impl, in.shape()[0]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_V) {
          out.push_back(V);
          if (!row_v) {
            if (out.back().dtype() == Type.ComplexFloat ||
                out.back().dtype() == Type.ComplexDouble) {
              out.back().permute_({1, 0}).contiguous_();
              out.back().Conj_();
            } else
              out.back().permute_({1, 0}).contiguous_();
          }
        }

        return out;

      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(in.device()));
        cytnx::linalg_internal::lii.cuEigh_ii[in.dtype()](in._impl->storage()._impl,
                                                          S._impl->storage()._impl,
                                                          V._impl->storage()._impl, in.shape()[0]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_V) {
          out.push_back(V);
          if (!row_v) {
            if (out.back().dtype() == Type.ComplexFloat ||
                out.back().dtype() == Type.ComplexDouble) {
              out.back().permute_({1, 0}).contiguous_();
              out.back().Conj_();
            } else
              out.back().permute_({1, 0}).contiguous_();
          }
        }

        return out;
  #else
        cytnx_error_msg(true, "[Eigh] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return std::vector<Tensor>();
  #endif
      }
    }

    // actual impls:
    static void Eigh_Dense_UT_internal(std::vector<cytnx::UniTensor> &outCyT, const UniTensor &Tin,
                                       const bool &is_V, const bool &row_v) {
      //[Note] outCyT must be empty!

      // DenseUniTensor:

      Tensor tmp;
      if (Tin.is_contiguous())
        tmp = Tin.get_block_();
      else {
        tmp = Tin.get_block();
        tmp.contiguous_();
      }

      std::vector<cytnx_uint64> tmps = tmp.shape();
      std::vector<cytnx_int64> oldshape(tmps.begin(), tmps.end());
      tmps.clear();
      std::vector<std::string> oldlabel = Tin.labels();

      // collapse as Matrix:
      cytnx_int64 rowdim = 1;
      for (cytnx_uint64 i = 0; i < Tin.rowrank(); i++) rowdim *= tmp.shape()[i];
      tmp.reshape_({rowdim, -1});

      std::vector<Tensor> outT = cytnx::linalg::Eigh(tmp, is_V, row_v);
      if (Tin.is_contiguous()) tmp.reshape_(oldshape);

      int t = 0;
      outCyT.resize(outT.size());

      // s
      cytnx::UniTensor &Cy_S = outCyT[t];
      cytnx::Bond newBond(outT[t].shape()[0]);

      Cy_S.Init({newBond, newBond}, {std::string("0"), std::string("1")},
                1,  // rowrank
                outT[t].dtype(), outT[t].device(),  // match the block that is inserted below
                true);  // is_diag
      Cy_S.put_block_(outT[t]);
      t++;
      if (is_V) {
        cytnx::UniTensor &Cy_U = outCyT[t];
        Cy_U.Init(outT[t], false, 1);  // Tin is a rowrank = 1 square UniTensor.
      }  // V
    }  // Eigh_Dense_UT_internal

    // Block-wise Hermitian eigendecomposition for a symmetric UniTensor. Handles both
    // BlockUniTensor (bosonic) and BlockFermionicUniTensor (fermionic), selected by the template
    // parameter BUT. For the fermionic case, sign-flipped blocks are negated to the physical
    // operator before the per-sector dense Eigh, and the resulting eigenvectors carry an all-false
    // signflip (they are already physical). For the bosonic case there are no sign flips and those
    // steps do nothing.
    template <class BUT>
    static void Eigh_BlockUT_internal(std::vector<cytnx::UniTensor> &outCyT, const UniTensor &Tin,
                                      const bool &is_V, const bool &row_v) {
      // outCyT must be empty and Tin must be checked with proper rowrank!

      // 1) getting the combineBond L and combineBond R for qnum list without grouping:
      //
      //   BDLeft -[ ]- BDRight
      //
      cytnx_error_msg(
        row_v && is_V,
        "[ERROR] Currently Eigh with row_v = true is not supported for symmetric UniTensors.%s",
        "\n");

      std::vector<bool> signflip;
      if constexpr (std::is_same_v<BUT, BlockFermionicUniTensor>)
        signflip = static_cast<BlockFermionicUniTensor *>(Tin._impl.get())->_signflip;
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
      // ->a initialize an empty shell of UniTensors!
      vec2d<cytnx_int64> aux_qnums;  // for sharing bond
      std::vector<cytnx_uint64> aux_degs;  // for sharing bond
      std::vector<Tensor> e_blocks;  // for eigenvalues

      vec2d<cytnx_uint64> v_itoi;  // for eigenvectors
      std::vector<Tensor> v_blocks;

      for (auto const &x : mgrp) {
        vec2d<cytnx_uint64> itoi_indicators(x.second.size());
        for (int i = 0; i < x.second.size(); i++) {
          itoi_indicators[i] = new_itoi[x.second[i]];
        }
        auto order = vec_sort(itoi_indicators, true);
        std::vector<Tensor> Tlist(itoi_indicators.size());
        std::vector<cytnx_int64> row_szs(order.size(), 1);
        cytnx_uint64 Rblk_dim = 0;
        cytnx_int64 tmp = -1;
        cytnx_int64 current_block;
        for (int i = 0; i < order.size(); i++) {
          current_block = x.second[order[i]];
          if (itoi_indicators[i][0] != tmp) {
            tmp = itoi_indicators[i][0];
            Rblk_dim++;
          }
          Tlist[i] = Tin.get_blocks_()[current_block];
          for (int j = 0; j < Tin.rowrank(); j++) {
            row_szs[i] *= Tlist[i].shape()[j];
          }
          bool flip = false;
          if constexpr (std::is_same_v<BUT, BlockFermionicUniTensor>)
            flip = signflip[current_block];
          if (flip) {
            Tlist[i] = -Tlist[i];  // copies Tensor
            Tlist[i].reshape_({row_szs[i], -1});
          } else {
            Tlist[i] = Tlist[i].reshape({row_szs[i], -1});  // copies Tensor
          }
        }
        cytnx_error_msg(Tlist.size() % Rblk_dim, "[Internal ERROR] Tlist is not complete!%s", "\n");
        // BTen is the big block!!
        cytnx_uint64 Cblk_dim = Tlist.size() / Rblk_dim;
        Tensor BTen = algo::_fx_Matric_combine(Tlist, Rblk_dim, Cblk_dim);
        //  Now we can perform linalg!
        aux_qnums.push_back(x.first);
        auto out = linalg::Eigh(BTen, is_V, row_v);
        aux_degs.push_back(out[0].shape()[0]);
        e_blocks.push_back(out[0]);

        if (is_V) {
          std::vector<cytnx_uint64> split_dims;
          for (int i = 0; i < Rblk_dim; i++) {
            split_dims.push_back(row_szs[i * Cblk_dim]);
          }
          std::vector<Tensor> blks;
          algo::Vsplit_(blks, out[1], split_dims);
          out[1] = Tensor();
          std::vector<cytnx_int64> new_shape(Tin.rowrank() + 1);
          new_shape.back() = -1;
          for (int ti = 0; ti < blks.size(); ti++) {
            v_blocks.push_back(blks[ti]);
            v_itoi.push_back(Tin.get_qindices(x.second[order[ti * Cblk_dim]]));

            // reshaping:
            for (int i = 0; i < Tin.rowrank(); i++) {
              new_shape[i] =
                Tin.bonds()[i]
                  .getDegeneracies()[Tin.get_qindices(x.second[order[ti * Cblk_dim]])[i]];
            }
            v_blocks.back().reshape_(new_shape);

            v_itoi.back()[Tin.rowrank()] = e_blocks.size() - 1;
            v_itoi.back().resize(Tin.rowrank() + 1);
          }
        }  // is_V
      }  // for each qcharge

      // process e:
      Bond Bd_aux = Bond(BD_IN, aux_qnums, aux_degs, Tin.syms());
      BUT *e_ptr = new BUT();
      e_ptr->Init(
        {Bd_aux, Bd_aux.redirect()}, {"_aux_L", "_aux_R"}, 1, Type.Double,
        Device.cpu,  // dtype, device are overwritten when the blocks are set; use defaults here
        true,  // is_diag!
        true);  // no_alloc!
      e_ptr->_blocks = e_blocks;
      UniTensor e;
      e._impl = boost::intrusive_ptr<UniTensor_base>(e_ptr);

      outCyT.push_back(e);

      if (is_V) {
        BUT *v_ptr = new BUT();
        for (int i = 0; i < Tin.rowrank(); i++) {
          v_ptr->_bonds.push_back(Tin.bonds()[i].clone());
          v_ptr->_labels.push_back(Tin.labels()[i]);
        }
        v_ptr->_bonds.push_back(Bd_aux.redirect());
        v_ptr->_labels.push_back("_aux_L");
        v_ptr->_rowrank = Tin.rowrank();
        v_ptr->_is_diag = false;
        v_ptr->_is_braket_form = v_ptr->_update_braket();
        v_ptr->_inner_to_outer_idx = v_itoi;
        v_ptr->_blocks = v_blocks;
        if constexpr (std::is_same_v<BUT, BlockFermionicUniTensor>)
          v_ptr->_signflip = std::vector<bool>(v_blocks.size(), false);
        UniTensor V;
        V._impl = boost::intrusive_ptr<UniTensor_base>(v_ptr);
        outCyT.push_back(V);
      }
    }  // Eigh_BlockUT_internal

    std::vector<cytnx::UniTensor> Eigh(const UniTensor &Tin, const bool &is_V, const bool &row_v) {
      // using rowrank to split the bond to form a matrix.
      cytnx_error_msg(Tin.rank() <= 1,
                      "[ERROR][Eigh] Input UniTensor should have rank>1, but rank is %d\n",
                      Tin.rank());
      cytnx_error_msg(Tin.rowrank() < 1,
                      "[ERROR][Eigh] Input UniTensor should have rowrank>0, but rowrank is %d\n",
                      Tin.rowrank());
      cytnx_error_msg(Tin.rowrank() >= Tin.rank(),
                      "[ERROR][Eigh] Input UniTensor should have rowrank<rank, but rowrank is %d "
                      "and rank is %d\n",
                      Tin.rowrank(), Tin.rank());
      cytnx_error_msg(Tin.is_diag(),
                      "[ERROR][Eigh] Input UniTensor is diagonal, so Eigh is trivial and not "
                      "supported. Use other manipulation.%s",
                      "\n");

      std::vector<UniTensor> outCyT;
      if (Tin.uten_type() == UTenType.Dense) {
        Eigh_Dense_UT_internal(outCyT, Tin, is_V, row_v);

      } else if (Tin.uten_type() == UTenType.Block) {
        Eigh_BlockUT_internal<BlockUniTensor>(outCyT, Tin, is_V, row_v);

      } else if (Tin.uten_type() == UTenType.BlockFermionic) {
        Eigh_BlockUT_internal<BlockFermionicUniTensor>(outCyT, Tin, is_V, row_v);

      } else {
        cytnx_error_msg(true, "[ERROR][Eigh] UniTensor type '%s' not supported\n",
                        Tin.uten_type_str().c_str());
      }  // ut type

      return outCyT;

    }  // Eigh

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
