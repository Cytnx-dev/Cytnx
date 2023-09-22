#include "linalg.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
#include "algo.hpp"
#include <iostream>
#include <vector>
using namespace std;
typedef cytnx::Accessor ac;

#ifdef BACKEND_TORCH
#else
  #include "../backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {

    std::vector<Tensor> Qr(const Tensor &Tin, const bool &is_tau) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[Qr] error, Qr can only operate on rank-2 Tensor.%s", "\n");
      // cytnx_error_msg(!Tin.is_contiguous(), "[Qr] error tensor must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      cytnx_uint64 n_tau = std::max(cytnx_uint64(1), std::min(Tin.shape()[0], Tin.shape()[1]));

      Tensor in = Tin.contiguous();
      if (Tin.dtype() > Type.Float) in = in.astype(Type.Double);

      // std::cout << n_singlu << std::endl;

      Tensor tau, Q, R, D;  // D is not used here.
      tau.Init({n_tau}, in.dtype(), in.device());
      tau.storage().set_zeros();  // if type is complex, S should be real
      R.Init({n_tau, Tin.shape()[1]}, in.dtype(), in.device());
      R.storage().set_zeros();

      if (Tin.device() == Device.cpu) {
        Q.Init({Tin.shape()[0], Tin.shape()[1]}, in.dtype(), in.device());
        Q.storage().set_zeros();
        cytnx::linalg_internal::lii.QR_ii[in.dtype()](
          in._impl->storage()._impl, Q._impl->storage()._impl, R._impl->storage()._impl,
          D._impl->storage()._impl, tau._impl->storage()._impl, in.shape()[0], in.shape()[1],
          false);

        std::vector<Tensor> out;
        if (in.shape()[0] < in.shape()[1]) Q = Q[{ac::all(), ac::range(0, in.shape()[0], 1)}];
        out.push_back(Q);
        out.push_back(R);

        if (is_tau) out.push_back(tau);

        return out;

      } else {
  #ifdef UNI_GPU
    #ifdef UNI_CUQUANTUM
        // cytnx_error_msg(true, "[Qr] error,%s", "Currently QR does not support CUDA.\n");

        checkCudaErrors(cudaSetDevice(in.device()));

        Q.Init({Tin.shape()[0], n_tau}, in.dtype(), in.device());
        Q.storage().set_zeros();

        cytnx::linalg_internal::lii.cuQuantumQr_ii[in.dtype()](
          in._impl->storage()._impl, Q._impl->storage()._impl, R._impl->storage()._impl,
          D._impl->storage()._impl, tau._impl->storage()._impl, in.shape()[0], in.shape()[1],
          false);

        std::vector<Tensor> out;
        out.push_back(Q);
        out.push_back(R);

        // if (is_tau) out.push_back(tau);
        cytnx_error_msg(is_tau, "[QR] Returning tau is currently not supported for cuQuantum QR.%s",
                        "\n");

        return out;
    #else
        cytnx_error_msg(true, "[QR] fatal error,%s",
                        "try to call the cuQuantum section without cuQuantum support.\n");
        return std::vector<Tensor>();
    #endif
  #else
        cytnx_error_msg(true, "[QR] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return std::vector<Tensor>();
  #endif
      }
    }

  }  // namespace linalg

}  // namespace cytnx

namespace cytnx {
  namespace linalg {

    void _qr_Dense_UT(std::vector<UniTensor> &outCyT, const UniTensor &Tin, const bool &is_tau) {
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

      vector<Tensor> outT = cytnx::linalg::Qr(tmp, is_tau);
      if (Tin.is_contiguous()) tmp.reshape_(oldshape);

      int t = 0;
      outCyT.resize(outT.size());

      string newlbl = "_aux_";

      // Q
      vector<cytnx_int64> Qshape;
      vector<string> Qlbl;
      for (int i = 0; i < Tin.rowrank(); i++) {
        Qshape.push_back(oldshape[i]);
        Qlbl.push_back(oldlabel[i]);
      }
      Qshape.push_back(-1);
      Qlbl.push_back(newlbl);
      outT[0].reshape_(Qshape);
      outCyT[0] = UniTensor(outT[0], false, Qshape.size() - 1);
      outCyT[0].set_labels(Qlbl);

      // R
      Qshape.clear();
      Qlbl.clear();
      Qshape.push_back(-1);
      Qlbl.push_back(newlbl);
      for (int i = Tin.rowrank(); i < Tin.rank(); i++) {
        Qshape.push_back(oldshape[i]);
        Qlbl.push_back(oldlabel[i]);
      }
      outT[1].reshape_(Qshape);
      outCyT[1] = UniTensor(outT[1], false, 1);
      outCyT[1].set_labels(Qlbl);

      // tau
      if (is_tau) {
        outCyT[2] = UniTensor(outT[2], false, 0);
      }

      if (Tin.is_tag()) {
        outCyT[0].tag();
        outCyT[1].tag();
        for (int i = 0; i < Tin.rowrank(); i++) {
          outCyT[0].bonds()[i].set_type(Tin.bonds()[i].type());
        }
        outCyT[0].bonds().back().set_type(cytnx::BD_BRA);
        outCyT[0]._impl->_is_braket_form = outCyT[0]._impl->_update_braket();

        outCyT[1].bonds()[0].set_type(cytnx::BD_KET);
        for (int i = 1; i < outCyT[1].rank(); i++) {
          outCyT[1].bonds()[i].set_type(Tin.bonds()[Tin.rowrank() + i - 1].type());
        }
        outCyT[1]._impl->_is_braket_form = outCyT[1]._impl->_update_braket();
      }
    };

    void _qr_Block_UT(std::vector<UniTensor> &outCyT, const UniTensor &Tin, const bool &is_tau) {
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

      std::vector<Tensor> tau_blocks;

      vec2d<cytnx_uint64> Q_itoi;  // for Q
      std::vector<Tensor> Q_blocks;

      vec2d<cytnx_uint64> R_itoi;  // for R
      std::vector<Tensor> R_blocks;

      cytnx_uint64 trcntr = 0;
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
          Tlist[i] = Tin.get_blocks()[x.second[order[i]]];
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
        auto out = linalg::Qr(BTen, is_tau);
        aux_degs.push_back(out[0].shape().back());

        // Q
        std::vector<cytnx_uint64> split_dims;
        for (int i = 0; i < Rblk_dim; i++) {
          split_dims.push_back(row_szs[i * Cblk_dim]);
        }
        std::vector<Tensor> blks;
        algo::Vsplit_(blks, out[0], split_dims);
        out[0] = Tensor();
        std::vector<cytnx_int64> new_shape(Tin.rowrank() + 1);
        new_shape.back() = -1;
        for (int ti = 0; ti < blks.size(); ti++) {
          Q_blocks.push_back(blks[ti]);
          Q_itoi.push_back(Tin.get_qindices(x.second[order[ti * Cblk_dim]]));

          // reshaping:
          for (int i = 0; i < Tin.rowrank(); i++) {
            new_shape[i] =
              Tin.bonds()[i].getDegeneracies()[Tin.get_qindices(x.second[order[ti * Cblk_dim]])[i]];
          }
          Q_blocks.back().reshape_(new_shape);

          Q_itoi.back()[Tin.rowrank()] = trcntr;
          Q_itoi.back().resize(Tin.rowrank() + 1);
        }

        // R
        split_dims.clear();
        for (int i = 0; i < Cblk_dim; i++) {
          split_dims.push_back(Tlist[i].shape().back());
        }
        blks.clear();
        algo::Hsplit_(blks, out[1], split_dims);
        out[1] = Tensor();

        new_shape.resize(Tin.rank() - Tin.rowrank() + 1);
        new_shape[0] = -1;
        for (int ti = 0; ti < blks.size(); ti++) {
          R_blocks.push_back(blks[ti]);
          auto &tpitoi = Tin.get_qindices(x.second[order[ti]]);
          R_itoi.push_back({trcntr});
          for (int i = Tin.rowrank(); i < Tin.rank(); i++) {
            R_itoi.back().push_back(tpitoi[i]);
          }

          // reshaping:
          for (int i = Tin.rowrank(); i < Tin.rank(); i++) {
            new_shape[i - Tin.rowrank() + 1] = Tin.bonds()[i].getDegeneracies()[tpitoi[i]];
          }
          R_blocks.back().reshape_(new_shape);
        }

        if (is_tau) {
          tau_blocks.push_back(out[2]);
        }

        trcntr++;

      }  // for each qcharge

      Bond Bd_aux = Bond(BD_IN, aux_qnums, aux_degs, Tin.syms());

      // process Q
      BlockUniTensor *Q_ptr = new BlockUniTensor();
      for (int i = 0; i < Tin.rowrank(); i++) {
        Q_ptr->_bonds.push_back(Tin.bonds()[i].clone());
        Q_ptr->_labels.push_back(Tin.labels()[i]);
      }
      Q_ptr->_bonds.push_back(Bd_aux.redirect());
      Q_ptr->_labels.push_back("_aux_");
      Q_ptr->_rowrank = Tin.rowrank();
      Q_ptr->_is_diag = false;
      Q_ptr->_is_braket_form = Q_ptr->_update_braket();
      Q_ptr->_inner_to_outer_idx = Q_itoi;
      Q_ptr->_blocks = Q_blocks;
      UniTensor Q;
      Q._impl = boost::intrusive_ptr<UniTensor_base>(Q_ptr);
      outCyT.push_back(Q);

      // process R:
      BlockUniTensor *R_ptr = new BlockUniTensor();
      R_ptr->_bonds.push_back(Bd_aux);
      R_ptr->_labels.push_back("_aux_");
      for (int i = Tin.rowrank(); i < Tin.rank(); i++) {
        R_ptr->_bonds.push_back(Tin.bonds()[i].clone());
        R_ptr->_labels.push_back(Tin.labels()[i]);
      }
      R_ptr->_rowrank = 1;
      R_ptr->_is_diag = false;
      R_ptr->_is_braket_form = R_ptr->_update_braket();
      R_ptr->_inner_to_outer_idx = R_itoi;
      R_ptr->_blocks = R_blocks;
      UniTensor R;
      R._impl = boost::intrusive_ptr<UniTensor_base>(R_ptr);
      outCyT.push_back(R);

      if (is_tau) {
        BlockUniTensor *tau_ptr = new BlockUniTensor();
        tau_ptr->Init({Bd_aux, Bd_aux.redirect()}, {"_tau_L", "_tau_R"}, 1, Type.Double,
                      Device.cpu,  // this two will be overwrite later, so doesnt matter.
                      true,  // is_diag!
                      true);  // no_alloc!
        tau_ptr->_blocks = tau_blocks;
        UniTensor tau;
        tau._impl = boost::intrusive_ptr<UniTensor_base>(tau_ptr);

        outCyT.push_back(tau);
      }
    };

    std::vector<UniTensor> Qr(const UniTensor &Tin, const bool &is_tau) {
      // using rowrank to split the bond to form a matrix.
      cytnx_error_msg(Tin.rowrank() < 1 || Tin.rank() == 1 || Tin.rowrank() == Tin.rank(),
                      "[QR][ERROR] QR for DenseUniTensor should have rank>1 and rank>rowrank>0%s",
                      "\n");

      std::vector<UniTensor> outCyT;
      if (Tin.uten_type() == UTenType.Dense) {
        _qr_Dense_UT(outCyT, Tin, is_tau);
        return outCyT;
      } else if (Tin.uten_type() == UTenType.Block) {
        _qr_Block_UT(outCyT, Tin, is_tau);
        return outCyT;

      } else {
        cytnx_error_msg(true, "[QR for sparse UniTensor is developling%s]", "\n");
      }
    };

  }  // namespace linalg
}  // namespace cytnx
#endif  // BACKEND_TORCH
