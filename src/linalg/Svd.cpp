#include "linalg.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
#include "algo.hpp"
#include <iostream>
#include <vector>
#include <string>
using namespace std;

#ifdef BACKEND_TORCH
#else
  #include "../backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {

    std::vector<Tensor> Svd(const Tensor &Tin, const bool &is_UvT) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[Svd] error, Svd can only operate on rank-2 Tensor.%s", "\n");
      // cytnx_error_msg(!Tin.is_contiguous(), "[Svd] error tensor must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      cytnx_uint64 n_singlu = std::max(cytnx_uint64(1), std::min(Tin.shape()[0], Tin.shape()[1]));

      Tensor in = Tin.contiguous();
      if (Tin.dtype() > Type.Float) in = in.astype(Type.Double);

      // std::cout << n_singlu << std::endl;

      Tensor U, S, vT;
      S.Init({n_singlu}, in.dtype() <= 2 ? in.dtype() + 2 : in.dtype(),
             in.device());  // if type is complex, S should be real
      // S.storage().set_zeros();
      if (is_UvT) {
        U.Init({in.shape()[0], n_singlu}, in.dtype(), in.device());
        // U.storage().set_zeros();
      }
      if (is_UvT) {
        vT.Init({n_singlu, in.shape()[1]}, in.dtype(), in.device());
        // vT.storage().set_zeros();
      }

      if (Tin.device() == Device.cpu) {
        // cytnx::linalg_internal::lii.Svd_ii[in.dtype()](
        //   in._impl->storage()._impl, U._impl->storage()._impl, vT._impl->storage()._impl,
        //   S._impl->storage()._impl, in.shape()[0], in.shape()[1]);
        cytnx::linalg_internal::lii.Sdd_ii[in.dtype()](
          in._impl->storage()._impl, U._impl->storage()._impl, vT._impl->storage()._impl,
          S._impl->storage()._impl, in.shape()[0], in.shape()[1]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_UvT) {
          out.push_back(U);
          out.push_back(vT);
        }

        return out;

      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(in.device()));
        cytnx::linalg_internal::lii.cuSvd_ii[in.dtype()](
          in._impl->storage()._impl, U._impl->storage()._impl, vT._impl->storage()._impl,
          S._impl->storage()._impl, in.shape()[0], in.shape()[1]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_UvT) {
          // cout << "Original:\n" << in << endl;
          // cout << "S:\n" << S << endl;
          // cout << "Recompose1!:\n" << Matmul(Matmul(U, Diag(S)), vT) << endl;
          // cout << "Recompose2!:\n"
          //      << Tensordot(Tensordot(U, Diag(S), {1}, {0}), vT, {1}, {0}) << endl;
          out.push_back(U);
          out.push_back(vT);
        }

        return out;
  #else
        cytnx_error_msg(true, "[Svd] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return std::vector<Tensor>();
  #endif
      }
    }

  }  // namespace linalg

}  // namespace cytnx

namespace cytnx {
  namespace linalg {

    // actual impls:
    void _svd_Dense_UT(std::vector<cytnx::UniTensor> &outCyT, const cytnx::UniTensor &Tin,
                       const bool &compute_uv) {
      //[Note] outCyT must be empty!

      // DenseUniTensor:
      // cout << "entry Dense UT" << endl;

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

      vector<Tensor> outT = cytnx::linalg::Svd(tmp, compute_uv);
      if (Tin.is_contiguous()) tmp.reshape_(oldshape);

      int t = 0;
      outCyT.resize(outT.size());

      // s
      cytnx::UniTensor &Cy_S = outCyT[t];
      cytnx::Bond newBond(outT[t].shape()[0]);

      Cy_S.Init({newBond, newBond}, {std::string("_aux_L"), std::string("_aux_R")}, 1, Type.Double,
                Device.cpu, true);  // it is just reference so no hurt to alias ^^

      // cout << "[AFTER INIT]" << endl;
      Cy_S.put_block_(outT[t]);
      t++;

      if (compute_uv) {
        cytnx::UniTensor &Cy_U = outCyT[t];
        vector<cytnx_int64> shapeU = vec_clone(oldshape, Tin.rowrank());
        shapeU.push_back(-1);
        outT[t].reshape_(shapeU);
        Cy_U.Init(outT[t], false, Tin.rowrank());
        vector<string> labelU = vec_clone(oldlabel, Tin.rowrank());
        labelU.push_back(Cy_S.labels()[0]);
        Cy_U.set_labels(labelU);
        t++;  // U
      }
      if (compute_uv) {
        cytnx::UniTensor &Cy_vT = outCyT[t];
        vector<cytnx_int64> shapevT(Tin.rank() - Tin.rowrank() + 1);
        shapevT[0] = -1;
        memcpy(&shapevT[1], &oldshape[Tin.rowrank()], sizeof(cytnx_int64) * (shapevT.size() - 1));

        outT[t].reshape_(shapevT);
        Cy_vT.Init(outT[t], false, 1);
        // cout << shapevT.size() << endl;
        vector<string> labelvT(shapevT.size());
        labelvT[0] = Cy_S.labels()[1];
        // memcpy(&labelvT[1], &oldlabel[Tin.rowrank()], sizeof(cytnx_int64) * (labelvT.size() -
        // 1));
        std::copy(oldlabel.begin() + Tin.rowrank(), oldlabel.end(), labelvT.begin() + 1);
        Cy_vT.set_labels(labelvT);
        t++;  // vT
      }
      // if tag, then update  the tagging informations
      if (Tin.is_tag()) {
        Cy_S.tag();
        t = 1;
        if (compute_uv) {
          cytnx::UniTensor &Cy_U = outCyT[t];
          Cy_U._impl->_is_tag = true;
          for (int i = 0; i < Cy_U.rowrank(); i++) {
            Cy_U.bonds()[i].set_type(Tin.bonds()[i].type());
          }
          Cy_U.bonds().back().set_type(cytnx::BD_BRA);
          Cy_U._impl->_is_braket_form = Cy_U._impl->_update_braket();
          t++;
        }
        if (compute_uv) {
          cytnx::UniTensor &Cy_vT = outCyT[t];
          Cy_vT._impl->_is_tag = true;
          Cy_vT.bonds()[0].set_type(cytnx::BD_KET);
          for (int i = 1; i < Cy_vT.rank(); i++) {
            Cy_vT.bonds()[i].set_type(Tin.bonds()[Tin.rowrank() + i - 1].type());
          }
          Cy_vT._impl->_is_braket_form = Cy_vT._impl->_update_braket();
          t++;
        }

      }  // if tag
    }

    void _svd_Block_UT(std::vector<cytnx::UniTensor> &outCyT, const cytnx::UniTensor &Tin,
                       const bool &compute_uv) {
      // outCyT must be empty and Tin must be checked with proper rowrank!

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
      // std::cout << BdRight << std::endl;
      // std::cout << strides << std::endl;

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
      // std::cout << new_itoi <<  std::endl;

      // 3) categorize:
      // key = qnum, val = list of block locations:
      std::map<std::vector<cytnx_int64>, std::vector<cytnx_int64>> mgrp;
      for (cytnx_uint64 b = 0; b < Tin.Nblocks(); b++) {
        mgrp[BdLeft.qnums()[new_itoi[b][0]]].push_back(b);
      }

      // 4) for each qcharge in key, combining the blocks into a big chunk!
      // ->a initialize an empty shell of UniTensor!
      vec2d<cytnx_int64> aux_qnums;  // for sharing bond
      std::vector<cytnx_uint64> aux_degs;  // forsharing bond
      std::vector<Tensor> S_blocks;

      vec2d<cytnx_uint64> U_itoi;  // for U
      std::vector<Tensor> U_blocks;

      vec2d<cytnx_uint64> vT_itoi;  // for vT
      std::vector<Tensor> vT_blocks;

      int tr;
      for (auto const &x : mgrp) {
        vec2d<cytnx_uint64> itoi_indicators(x.second.size());
        // cout << x.second.size() << "-------" << endl;
        for (int i = 0; i < x.second.size(); i++) {
          itoi_indicators[i] = new_itoi[x.second[i]];
          // std::cout << new_itoi[x.second[i]] << std::endl;
        }
        auto order = vec_sort(itoi_indicators, true);
        std::vector<Tensor> Tlist(itoi_indicators.size());
        std::vector<cytnx_int64> row_szs(order.size(), 1);
        cytnx_uint64 Rblk_dim = 0;
        cytnx_int64 tmp = -1;
        for (int i = 0; i < order.size(); i++) {
          if (itoi_indicators[i][0] != tmp) {
            tmp = itoi_indicators[i][0];
            Rblk_dim++;
          }
          Tlist[i] = Tin.get_blocks_()[x.second[order[i]]];
          for (int j = 0; j < Tin.rowrank(); j++) {
            row_szs[i] *= Tlist[i].shape()[j];
          }
          Tlist[i] = Tlist[i].reshape({row_szs[i], -1});
        }
        cytnx_error_msg(Tlist.size() % Rblk_dim, "[Internal ERROR] Tlist is not complete!%s", "\n");
        // BTen is the big block!!
        cytnx_uint64 Cblk_dim = Tlist.size() / Rblk_dim;
        Tensor BTen = algo::_fx_Matric_combine(Tlist, Rblk_dim, Cblk_dim);

        // Now we can perform linalg!
        aux_qnums.push_back(x.first);
        auto out = linalg::Svd(BTen, compute_uv);
        aux_degs.push_back(out[0].shape()[0]);
        S_blocks.push_back(out[0]);
        tr = 1;

        if (compute_uv) {
          // std::cout << row_szs << std::endl;
          // std::cout << out[tr].shape() << std::endl;
          std::vector<cytnx_uint64> split_dims;
          for (int i = 0; i < Rblk_dim; i++) {
            split_dims.push_back(row_szs[i * Cblk_dim]);
          }
          std::vector<Tensor> blks;
          algo::Vsplit_(blks, out[tr], split_dims);
          out[tr] = Tensor();
          std::vector<cytnx_int64> new_shape(Tin.rowrank() + 1);
          new_shape.back() = -1;
          for (int ti = 0; ti < blks.size(); ti++) {
            U_blocks.push_back(blks[ti]);
            U_itoi.push_back(Tin.get_qindices(x.second[order[ti * Cblk_dim]]));

            // reshaping:
            for (int i = 0; i < Tin.rowrank(); i++) {
              new_shape[i] =
                Tin.bonds()[i]
                  .getDegeneracies()[Tin.get_qindices(x.second[order[ti * Cblk_dim]])[i]];
            }
            U_blocks.back().reshape_(new_shape);

            U_itoi.back()[Tin.rowrank()] = S_blocks.size() - 1;
            U_itoi.back().resize(Tin.rowrank() + 1);
          }
          tr++;
        }  // is_U

        if (compute_uv) {
          std::vector<cytnx_uint64> split_dims;
          for (int i = 0; i < Cblk_dim; i++) {
            split_dims.push_back(Tlist[i].shape().back());
          }
          std::vector<Tensor> blks;
          algo::Hsplit_(blks, out[tr], split_dims);
          out[tr] = Tensor();

          std::vector<cytnx_int64> new_shape(Tin.rank() - Tin.rowrank() + 1);
          new_shape[0] = -1;
          for (int ti = 0; ti < blks.size(); ti++) {
            vT_blocks.push_back(blks[ti]);
            auto &tpitoi = Tin.get_qindices(x.second[order[ti]]);
            vT_itoi.push_back({S_blocks.size() - 1});
            for (int i = Tin.rowrank(); i < Tin.rank(); i++) {
              vT_itoi.back().push_back(tpitoi[i]);
            }

            // reshaping:
            for (int i = Tin.rowrank(); i < Tin.rank(); i++) {
              new_shape[i - Tin.rowrank() + 1] = Tin.bonds()[i].getDegeneracies()[tpitoi[i]];
            }
            vT_blocks.back().reshape_(new_shape);
          }

          tr++;
        }  // is_vT
      }

      // process S:
      Bond Bd_aux = Bond(BD_IN, aux_qnums, aux_degs, Tin.syms());
      BlockUniTensor *S_ptr = new BlockUniTensor();
      S_ptr->Init({Bd_aux, Bd_aux.redirect()}, {"_aux_L", "_aux_R"}, 1, Type.Double,
                  Device.cpu,  // this two will be overwrite later, so doesnt matter.
                  true,  // is_diag!
                  true);  // no_alloc!
      S_ptr->_blocks = S_blocks;
      UniTensor S;
      S._impl = boost::intrusive_ptr<UniTensor_base>(S_ptr);

      outCyT.push_back(S);

      if (compute_uv) {
        BlockUniTensor *U_ptr = new BlockUniTensor();
        for (int i = 0; i < Tin.rowrank(); i++) {
          U_ptr->_bonds.push_back(Tin.bonds()[i].clone());
          U_ptr->_labels.push_back(Tin.labels()[i]);
        }
        U_ptr->_bonds.push_back(Bd_aux.redirect());
        U_ptr->_labels.push_back("_aux_L");
        U_ptr->_rowrank = Tin.rowrank();
        U_ptr->_is_diag = false;
        U_ptr->_is_braket_form = U_ptr->_update_braket();
        U_ptr->_inner_to_outer_idx = U_itoi;
        U_ptr->_blocks = U_blocks;
        UniTensor U;
        U._impl = boost::intrusive_ptr<UniTensor_base>(U_ptr);
        outCyT.push_back(U);
      }

      if (compute_uv) {
        BlockUniTensor *vT_ptr = new BlockUniTensor();
        vT_ptr->_bonds.push_back(Bd_aux);
        vT_ptr->_labels.push_back("_aux_R");

        for (int i = Tin.rowrank(); i < Tin.rank(); i++) {
          vT_ptr->_bonds.push_back(Tin.bonds()[i].clone());
          vT_ptr->_labels.push_back(Tin.labels()[i]);
        }
        vT_ptr->_rowrank = 1;
        vT_ptr->_is_diag = false;
        vT_ptr->_is_braket_form = vT_ptr->_update_braket();
        vT_ptr->_inner_to_outer_idx = vT_itoi;
        vT_ptr->_blocks = vT_blocks;
        UniTensor vT;
        vT._impl = boost::intrusive_ptr<UniTensor_base>(vT_ptr);
        outCyT.push_back(vT);
      }

    }  // _svd_Block_UT

    std::vector<cytnx::UniTensor> Svd(const cytnx::UniTensor &Tin, const bool &is_UvT) {
      // using rowrank to split the bond to form a matrix.
      cytnx_error_msg(Tin.rowrank() < 1 || Tin.rank() == 1,
                      "[Svd][ERROR] Svd for UniTensor should have rank>1 and rowrank>0%s", "\n");

      cytnx_error_msg(Tin.is_diag(),
                      "[Svd][ERROR] Svd for diagonal UniTensor is trivial and currently not "
                      "support. Use other manipulation.%s",
                      "\n");

      std::vector<UniTensor> outCyT;
      if (Tin.uten_type() == UTenType.Dense) {
        _svd_Dense_UT(outCyT, Tin, is_UvT);

      } else if (Tin.uten_type() == UTenType.Block) {
        _svd_Block_UT(outCyT, Tin, is_UvT);

      } else {
        cytnx_error_msg(true, "[ERROR] only support svd for Dense and Block UniTensor.%s", "\n");

      }  // is block form ?

      return outCyT;

    };  // Svd

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
