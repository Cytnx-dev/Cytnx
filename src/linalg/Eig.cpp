#include "linalg.hpp"
#include "algo.hpp"
#include <iostream>
#include "Tensor.hpp"
using namespace std;

#ifdef BACKEND_TORCH
#else

  #include "../backend/linalg_internal_interface.hpp"
namespace cytnx {
  namespace linalg {
    std::vector<Tensor> Eig(const Tensor &Tin, const bool &is_V, const bool &row_v) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[Eig] error, Eig can only operate on rank-2 Tensor.%s", "\n");
      // cytnx_error_msg(!Tin.is_contiguous(), "[Eig] error tensor must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1],
                      "[Eig] error, Eig should accept a square matrix%s", "\n");

      Tensor in = Tin.contiguous();
      if (Tin.dtype() > Type.Float)
        in = in.astype(Type.ComplexDouble);
      else {
        if (Tin.dtype() == Type.Float)
          in = in.astype(Type.ComplexFloat);
        else if (Tin.dtype() == Type.Double)
          in = in.astype(Type.ComplexDouble);
      }

      Tensor S, V;
      S.Init({in.shape()[0]}, in.dtype() % 2 == 1 ? Type.ComplexDouble : Type.ComplexFloat,
             in.device());  // S should always be double!!

      if (is_V) {
        V.Init(in.shape(), in.dtype(), in.device()); /*V.storage().set_zeros();*/
      }

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Eig_ii[in.dtype()](in._impl->storage()._impl,
                                                       S._impl->storage()._impl,
                                                       V._impl->storage()._impl, in.shape()[0]);

        std::vector<Tensor> out;
        out.push_back(S);
        // std::cout << "[Eig][V]" << std::endl;
        // std::cout << V << std::endl;
        if (is_V) {
          out.push_back(V);
          if (!row_v) {
            if (out.back().dtype() == Type.ComplexFloat ||
                out.back().dtype() == Type.ComplexDouble) {
              out.back().permute_({1, 0}).contiguous_();
              // std::cout << out.back();
              out.back().Conj_();
              // std::cout << out.back();
              // std::cout << "ok" << std::endl;
            } else
              out.back().permute_({1, 0}).contiguous_();
          }
        }

        return out;

      } else {
  #ifdef UNI_GPU
        /*
        checkCudaErrors(cudaSetDevice(in.device()));
        cytnx::linalg_internal::lii.cuEig_ii[in.dtype()](in._impl->storage()._impl,
                                                S._impl->storage()._impl,
                                                V._impl->storage()._impl,
                                                in.shape()[0]);

        std::vector<Tensor> out;
        out.push_back(S);
        if(is_V){
            out.push_back(V);
            if(!row_v) out.back().permute_({1,0}).contiguous_();
        }
        */
        cytnx_error_msg(true, "[ERROR]currently Eig for non-symmetric matrix is not supported.%s",
                        "\n");
        return std::vector<Tensor>();
  #else
        cytnx_error_msg(true, "[Eig] fatal error,%s",
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
    void _Eig_Dense_UT(std::vector<cytnx::UniTensor> &outCyT, const UniTensor &Tin,
                       const bool &is_V, const bool &row_v) {
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

      vector<Tensor> outT = cytnx::linalg::Eig(tmp, is_V, row_v);
      if (Tin.is_contiguous()) tmp.reshape_(oldshape);

      int t = 0;
      outCyT.resize(outT.size());

      // s
      cytnx::UniTensor &Cy_S = outCyT[t];
      cytnx::Bond newBond(outT[t].shape()[0]);

      Cy_S.Init({newBond, newBond}, {std::string("0"), std::string("1")}, 1, Type.Double,
                Device.cpu, true);  // it is just reference so no hurt to alias ^^. All eigvals are
                                    // real for eigh so Type.Double.

      Cy_S.put_block_(outT[t]);
      t++;
      if (is_V) {
        cytnx::UniTensor &Cy_U = outCyT[t];
        Cy_U.Init(outT[t], false, 1);  // Tin is a rowrank = 1 square UniTensor.
      }  // V
    }

    void _Eig_Block_UT(std::vector<cytnx::UniTensor> &outCyT, const UniTensor &Tin,
                       const bool &is_V, const bool &row_v) {
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
      std::vector<Tensor> e_blocks;  // for eigenvalues

      vec2d<cytnx_uint64> v_itoi;  // for eigen vectors
      std::vector<Tensor> v_blocks;

      // vec2d<cytnx_uint64> vT_itoi;  // for vT
      // std::vector<Tensor> vT_blocks;

      for (auto const &x : mgrp) {
        vec2d<cytnx_uint64> itoi_indicators(x.second.size());
        // cout << x.second.size() << "-------" << endl; //
        for (int i = 0; i < x.second.size(); i++) {
          itoi_indicators[i] = new_itoi[x.second[i]];
          // std::cout << new_itoi[x.second[i]] << std::endl; //
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
        // std::cout << BTen;
        //  Now we can perform linalg!
        aux_qnums.push_back(x.first);
        auto out = linalg::Eig(BTen, is_V, row_v);
        aux_degs.push_back(out[0].shape()[0]);
        e_blocks.push_back(out[0]);

        if (is_V) {
          // std::cout << row_szs << std::endl;
          // std::cout << out[tr].shape() << std::endl;
          std::vector<cytnx_uint64> split_dims;
          for (int i = 0; i < Rblk_dim; i++) {
            split_dims.push_back(row_szs[i * Cblk_dim]);
          }
          std::vector<Tensor> blks;
          // std::cout<<out[1];
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
      }

      // process e:
      Bond Bd_aux = Bond(BD_IN, aux_qnums, aux_degs, Tin.syms());
      BlockUniTensor *e_ptr = new BlockUniTensor();
      e_ptr->Init({Bd_aux, Bd_aux.redirect()}, {"_aux_L", "_aux_R"}, 1, Type.Double,
                  Device.cpu,  // this two will be overwrite later, so doesnt matter.
                  true,  // is_diag!
                  true);  // no_alloc!
      e_ptr->_blocks = e_blocks;
      UniTensor e;
      e._impl = boost::intrusive_ptr<UniTensor_base>(e_ptr);

      outCyT.push_back(e);

      if (is_V) {
        BlockUniTensor *v_ptr = new BlockUniTensor();
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
        UniTensor V;
        V._impl = boost::intrusive_ptr<UniTensor_base>(v_ptr);
        outCyT.push_back(V);
      }
    }

    std::vector<cytnx::UniTensor> Eig(const UniTensor &Tin, const bool &is_V, const bool &row_v) {
      // using rowrank to split the bond to form a matrix.
      cytnx_error_msg(Tin.rowrank() < 1 || Tin.rank() == 1,
                      "[Eig][ERROR] Eig for UniTensor should have rank>1 and rowrank>0%s", "\n");

      cytnx_error_msg(Tin.is_diag(),
                      "[Eig][ERROR] Eig for diagonal UniTensor is trivial and currently not "
                      "support. Use other manipulation.%s",
                      "\n");

      std::vector<UniTensor> outCyT;
      if (Tin.uten_type() == UTenType.Dense) {
        _Eig_Dense_UT(outCyT, Tin, is_V, row_v);

      } else if (Tin.uten_type() == UTenType.Block) {
        _Eig_Block_UT(outCyT, Tin, is_V, row_v);
      } else {
        cytnx_error_msg(true,
                        "[ERROR] Eig, unsupported type of UniTensor only support (Dense, Block). "
                        "something wrong internal%s",
                        "\n");
      }  // ut type

      return outCyT;

    };  // Eig

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
