#include <vector>

#include "Accessor.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
#include "algo.hpp"
#include "linalg.hpp"

#ifdef BACKEND_TORCH
#else

  #include "backend/linalg_internal_interface.hpp"

  #ifdef UNI_GPU
    #ifdef UNI_CUQUANTUM
      #include "backend/linalg_internal_gpu/cuQuantumGeSvd_internal.hpp"
    #endif
  #endif

namespace cytnx {
  namespace linalg {
    std::vector<Tensor> Gesvd_truncate(const Tensor &Tin, const cytnx_uint64 &keepdim,
                                       const double &err, const bool &is_U, const bool &is_vT,
                                       const unsigned int &return_err, const cytnx_uint64 &mindim) {
      // check input arguments
      cytnx_error_msg(mindim < 0, "[ERROR][Gesvd_truncate] mindim must be >=1.%s", "\n");
      cytnx_error_msg(keepdim < 1, "[ERROR][Gesvd_truncate] keepdim must be >=1.%s", "\n");
      cytnx_error_msg(return_err < 0, "[ERROR][Gesvd_truncate] return_err cannot be negative%s",
                      "\n");
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[Gesvd_truncate] can only operate on rank-2 Tensor.%s", "\n");

      if (Tin.device() == Device.cpu) {
        std::vector<Tensor> tmps = Gesvd(Tin, is_U, is_vT);
        Tensor terr({1}, Tin.dtype(), Tin.device());

        cytnx::linalg_internal::lii.memcpyTruncation_ii[Tin.dtype()](
          tmps[1], tmps[2], tmps[0], terr, keepdim, err, is_U, is_vT, return_err, mindim);

        std::vector<Tensor> outT;
        outT.push_back(tmps[0]);
        if (is_U) outT.push_back(tmps[1]);
        if (is_vT) outT.push_back(tmps[2]);
        if (return_err) outT.push_back(terr);

        return outT;

      } else {
  #ifdef UNI_GPU
    #ifdef UNI_CUQUANTUM
        Tensor in = Tin.contiguous();

        // cytnx_uint64 n_singlu = std::min(keepdim, std::min(Tin.shape()[0], Tin.shape()[1]));
        cytnx_uint64 n_singlu = std::max(cytnx_uint64(1), std::min(Tin.shape()[0], Tin.shape()[1]));
        // if (Tin.dtype() > Type.Float) in = in.astype(Type.Double);
        // prepare U, S, vT
        Tensor U, S, vT, terr;
        S.Init({n_singlu}, in.dtype() <= 2 ? in.dtype() + 2 : in.dtype(),
               in.device());  // if type is complex, S should be real
        U.Init({in.shape()[0], n_singlu}, in.dtype(), in.device());
        vT.Init({n_singlu, in.shape()[1]}, in.dtype(), in.device());
        terr.Init({1}, in.dtype(), in.device());

        cytnx::linalg_internal::lii.cuQuantumGeSvd_ii[in.dtype()](in, keepdim, err, return_err, U,
                                                                  S, vT, terr);

        cytnx::linalg_internal::lii.cudaMemcpyTruncation_ii[in.dtype()](
          U, vT, S, terr, keepdim, err, is_U, is_vT, return_err, mindim);

        std::vector<Tensor> outT;
        outT.push_back(S);
        if (is_U) outT.push_back(U);
        if (is_vT) outT.push_back(vT);
        if (return_err) outT.push_back(terr);

        return outT;

    #else
        std::vector<Tensor> tmps = Gesvd(Tin, is_U, is_vT);
        Tensor terr({1}, Tin.dtype(), Tin.device());

        cytnx::linalg_internal::lii.cudaMemcpyTruncation_ii[Tin.dtype()](
          tmps[1], tmps[2], tmps[0], terr, keepdim, err, is_U, is_vT, return_err, mindim);

        std::vector<Tensor> outT;
        outT.push_back(tmps[0]);
        if (is_U) outT.push_back(tmps[1]);
        if (is_vT) outT.push_back(tmps[2]);
        if (return_err) outT.push_back(terr);

        return outT;
    #endif
  #else
        cytnx_error_msg(
          true, "[Error][Gesvd_truncate] Trying to call the gpu section without CUDA support%s",
          "\n");
        return std::vector<Tensor>();
  #endif
      }
    }
  }  // namespace linalg
}  // namespace cytnx

namespace cytnx {
  namespace linalg {
    using namespace std;
    typedef Accessor ac;

    void _gesvd_truncate_Dense_UT(std::vector<UniTensor> &outCyT, const cytnx::UniTensor &Tin,
                                  const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                                  const bool &is_vT, const unsigned int &return_err,
                                  const cytnx_uint64 &mindim) {
      // DenseUniTensor:
      cytnx_uint64 keep_dim = keepdim;

      Tensor tmp = Tin.get_block_().contiguous();
      // if(Tin.is_contiguous()) tmp = Tin.get_block_();
      // else{ tmp = Tin.get_block(); tmp.contiguous_();}

      vector<cytnx_uint64> tmps = tmp.shape();
      vector<cytnx_int64> oldshape(tmps.begin(), tmps.end());
      tmps.clear();
      vector<string> oldlabel = Tin.labels();

      // collapse as Matrix:
      cytnx_int64 rowdim = 1;
      for (cytnx_uint64 i = 0; i < Tin.rowrank(); i++) rowdim *= tmp.shape()[i];
      tmp = tmp.reshape({rowdim, -1});

      vector<Tensor> outT =
        cytnx::linalg::Gesvd_truncate(tmp, keepdim, err, is_U, is_vT, return_err, mindim);

      // if(Tin.is_contiguous()) tmp.reshape_(oldshape);

      int t = 0;
      outCyT.resize(outT.size());

      // s
      // cytnx_error_msg(keepdim>outT[t].shape()[0],"[ERROR][Gesvd_truncate] keepdim should <=
      // dimension of singular tensor%s","\n");

      cytnx::UniTensor &Cy_S = outCyT[t];
      cytnx::Bond newBond(outT[0].shape()[0]);
      Cy_S.Init({newBond, newBond}, {string("_aux_L"), string("_aux_R")}, 1, Type.Double,
                Tin.device(),
                true);  // it is just reference so no hurt to alias ^^
      Cy_S.put_block_(outT[t]);
      t++;

      if (is_U) {
        cytnx::UniTensor &Cy_U = outCyT[t];
        // shape
        cytnx_error_msg(Tin.rowrank() > oldshape.size(),
                        "[ERROR] The rowrank of the input unitensor is larger than the rank of the "
                        "contained tensor.%s",
                        "\n");
        vector<cytnx_int64> shapeU(oldshape.begin(), oldshape.begin() + Tin.rowrank());
        shapeU.push_back(-1);

        outT[t].reshape_(shapeU);

        Cy_U.Init(outT[t], false, Tin.rowrank());
        vector<string> labelU(oldlabel.begin(), oldlabel.begin() + Tin.rowrank());
        labelU.push_back(Cy_S.labels()[0]);
        Cy_U.set_labels(labelU);
        t++;  // U
      }

      if (is_vT) {
        cytnx::UniTensor &Cy_vT = outCyT[t];

        // shape
        vector<cytnx_int64> shapevT(Tin.rank() - Tin.rowrank() + 1);
        shapevT[0] = -1;
        memcpy(&shapevT[1], &oldshape[Tin.rowrank()], sizeof(cytnx_int64) * (shapevT.size() - 1));

        outT[t].reshape_(shapevT);

        Cy_vT.Init(outT[t], false, 1);
        vector<string> labelvT(shapevT.size());
        labelvT[0] = Cy_S.labels()[1];
        std::copy(oldlabel.begin() + Tin.rowrank(), oldlabel.end(), labelvT.begin() + 1);
        Cy_vT.set_labels(labelvT);
        t++;  // vT
      }

      // if tag, then update  the tagging informations
      if (Tin.is_tag()) {
        Cy_S.tag();
        t = 1;
        if (is_U) {
          cytnx::UniTensor &Cy_U = outCyT[t];
          Cy_U._impl->_is_tag = true;
          for (int i = 0; i < Cy_U.rowrank(); i++) {
            Cy_U.bonds()[i].set_type(Tin.bonds()[i].type());
          }
          Cy_U.bonds().back().set_type(cytnx::BD_BRA);
          Cy_U._impl->_is_braket_form = Cy_U._impl->_update_braket();
          t++;
        }
        if (is_vT) {
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

      if (return_err) outCyT.back().Init(outT.back(), false, 0);
    };  // svdt Dense

    void _gesvd_truncate_Block_UT(std::vector<UniTensor> &outCyT, const cytnx::UniTensor &Tin,
                                  const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                                  const bool &is_vT, const unsigned int &return_err,
                                  const cytnx_uint64 &mindim) {
      cytnx_uint64 keep_dim = keepdim;

      outCyT = linalg::Gesvd(Tin, is_U, is_vT);

      // process truncation:
      // 1) concate all S vals from all blk
      Tensor Sall = outCyT[0].get_block_(0);
      for (int i = 1; i < outCyT[0].Nblocks(); i++) {
        Sall = algo::Concatenate(Sall, outCyT[0].get_block_(i));
      }
      Sall = algo::Sort(Sall);  // all singular values, starting from the smallest

      // 2) get the minimum S value based on the args input.
      Scalar Smin;
      cytnx_uint64 smidx;
      cytnx_uint64 Sshape = Sall.shape()[0];
      if (keep_dim < Sshape) {
        smidx = Sshape - keep_dim;
        Smin = Sall.storage()(smidx);
      } else {
        keep_dim = Sshape;
        smidx = 0;
        Smin = Sall.storage()(0);
      }
      while ((Smin < err) and (keep_dim > (mindim < 1 ? 1 : mindim))) {
        // at least one singular value is always kept!
        keep_dim--;
        // if (keep_dim == 0) break;
        smidx++;
        Smin = Sall.storage()(smidx);
      }

      // traversal each block and truncate!
      UniTensor &S = outCyT[0];
      std::vector<cytnx_uint64> new_dims;  // keep_dims for each block!
      std::vector<cytnx_int64> keep_dims;
      keep_dims.reserve(S.Nblocks());
      std::vector<cytnx_int64> new_qid;
      new_qid.reserve(S.Nblocks());

      std::vector<std::vector<cytnx_uint64>> new_itoi;  // assume S block is in same order as qnum:
      std::vector<cytnx_uint64> to_be_removed;

      cytnx_uint64 tot_dim = 0;
      cytnx_uint64 cnt = 0;
      for (int b = 0; b < S.Nblocks(); b++) {
        Storage stmp = S.get_block_(b).storage();
        cytnx_int64 kdim = 0;
        for (int i = stmp.size(); i > 0; i--) {
          if (stmp(i - 1) >= Smin) {
            kdim = i;
            break;
          }
        }
        keep_dims.push_back(kdim);
        if (kdim == 0) {
          to_be_removed.push_back(b);
          new_qid.push_back(-1);

        } else {
          new_qid.push_back(new_dims.size());
          new_itoi.push_back({new_dims.size(), new_dims.size()});
          new_dims.push_back(kdim);
          tot_dim += kdim;
          if (kdim != S.get_blocks_()[b].shape()[0])
            S.get_blocks_()[b] = S.get_blocks_()[b].get({Accessor::range(0, kdim)});
        }
      }

      // remove:
      // vec_erase_(S.get_itoi(),to_be_removed);
      S.get_itoi() = new_itoi;
      vec_erase_(S.get_blocks_(), to_be_removed);
      vec_erase_(S.bonds()[0].qnums(), to_be_removed);
      S.bonds()[0]._impl->_degs = new_dims;
      S.bonds()[0]._impl->_dim = tot_dim;
      S.bonds()[1] = S.bonds()[0].redirect();

      int t = 1;
      if (is_U) {
        UniTensor &U = outCyT[t];
        to_be_removed.clear();
        U.bonds().back() = S.bonds()[1].clone();
        std::vector<Accessor> acs(U.rank());
        for (int i = 0; i < U.rowrank(); i++) acs[i] = Accessor::all();

        for (int b = 0; b < U.Nblocks(); b++) {
          if (keep_dims[U.get_qindices(b).back()] == 0)
            to_be_removed.push_back(b);
          else {
            /// process blocks:
            if (keep_dims[U.get_qindices(b).back()] != U.get_blocks_()[b].shape().back()) {
              acs.back() = Accessor::range(0, keep_dims[U.get_qindices(b).back()]);
              U.get_blocks_()[b] = U.get_blocks_()[b].get(acs);
            }

            // change to new qindices:
            U.get_qindices(b).back() = new_qid[U.get_qindices(b).back()];
          }
        }
        vec_erase_(U.get_itoi(), to_be_removed);
        vec_erase_(U.get_blocks_(), to_be_removed);

        t++;
      }

      if (is_vT) {
        UniTensor &vT = outCyT[t];
        to_be_removed.clear();
        vT.bonds().front() = S.bonds()[0].clone();
        std::vector<Accessor> acs(vT.rank());
        for (int i = 1; i < vT.rank(); i++) acs[i] = Accessor::all();

        for (int b = 0; b < vT.Nblocks(); b++) {
          if (keep_dims[vT.get_qindices(b)[0]] == 0)
            to_be_removed.push_back(b);
          else {
            /// process blocks:
            if (keep_dims[vT.get_qindices(b)[0]] != vT.get_blocks_()[b].shape()[0]) {
              acs[0] = Accessor::range(0, keep_dims[vT.get_qindices(b)[0]]);
              vT.get_blocks_()[b] = vT.get_blocks_()[b].get(acs);
            }
            // change to new qindices:
            vT.get_qindices(b)[0] = new_qid[vT.get_qindices(b)[0]];
          }
        }
        vec_erase_(vT.get_itoi(), to_be_removed);
        vec_erase_(vT.get_blocks_(), to_be_removed);
        t++;
      }

      // handle return_err!
      if (return_err == 1) {
        outCyT.push_back(UniTensor(Tensor({1}, Smin.dtype())));
        outCyT.back().get_block_().storage().at(0) = Smin;
      } else if (return_err) {
        outCyT.push_back(UniTensor(Sall.get({Accessor::tilend(smidx)})));
      }
    }  // _gesvd_truncate_Block_UT

    void _gesvd_truncate_BlockFermionic_UT(std::vector<UniTensor> &outCyT,
                                           const cytnx::UniTensor &Tin, const cytnx_uint64 &keepdim,
                                           const double &err, const bool &is_U, const bool &is_vT,
                                           const unsigned int &return_err,
                                           const unsigned int &mindim) {
      //[9 Oct 2024] This is a copy from _gesvd_truncate_Block_UT;
      // TODOfermionic: remove signs if blocks are deleted
      cytnx_error_msg(true,
                      "[ERROR][_gesvd_truncate_BlockFermionic_UT] not implemented yet. The "
                      "signflips need to be removed if blocks are deleted in the truncation.%s",
                      "\n") cytnx_uint64 keep_dim = keepdim;

      outCyT = linalg::Gesvd(Tin, is_U, is_vT);

      // process truncate:
      // 1) concate all s vals from all blk
      Tensor Sall = outCyT[0].get_block_(0);
      for (int i = 1; i < outCyT[0].Nblocks(); i++) {
        Sall = algo::Concatenate(Sall, outCyT[0].get_block_(i));
      }
      Sall = algo::Sort(Sall);

      // 2) get the minimum base on the args input.
      Scalar Smin;
      cytnx_uint64 smidx;
      if (keep_dim < Sall.shape()[0]) {
        smidx = Sall.shape()[0] - keep_dim;
        Smin = Sall.storage()(smidx);
        while ((Smin < err) and keep_dim - 1 > mindim) {
          keep_dim -= 1;
          if (keep_dim == 0) break;
          smidx = Sall.shape()[0] - keep_dim;
          Smin = Sall.storage()(smidx);
        }

      } else {
        keep_dim = Sall.shape()[0];
        Smin = Sall.storage()(0);
        smidx = 0;
        while ((Smin < err)) {
          keep_dim -= 1;
          if (keep_dim == 0) break;
          smidx = Sall.shape()[0] - keep_dim;
          Smin = Sall.storage()(smidx);
        }
      }

      // traversal each block and truncate!
      UniTensor &S = outCyT[0];
      std::vector<cytnx_uint64> new_dims;  // keep_dims for each block!
      std::vector<cytnx_int64> keep_dims;
      keep_dims.reserve(S.Nblocks());
      std::vector<cytnx_int64> new_qid;
      new_qid.reserve(S.Nblocks());

      std::vector<std::vector<cytnx_uint64>> new_itoi;  // assume S block is in same order as qnum:
      std::vector<cytnx_uint64> to_be_removed;

      cytnx_uint64 tot_dim = 0;
      cytnx_uint64 cnt = 0;
      for (int b = 0; b < S.Nblocks(); b++) {
        Storage stmp = S.get_block_(b).storage();
        cytnx_int64 kdim = 0;
        for (int i = stmp.size() - 1; i >= 0; i--) {
          if (stmp(i) >= Smin) {
            kdim = i + 1;
            break;
          }
        }
        keep_dims.push_back(kdim);
        if (kdim == 0) {
          to_be_removed.push_back(b);
          new_qid.push_back(-1);

        } else {
          new_qid.push_back(new_dims.size());
          new_itoi.push_back({new_dims.size(), new_dims.size()});
          new_dims.push_back(kdim);
          tot_dim += kdim;
          if (kdim != S.get_blocks_()[b].shape()[0])
            S.get_blocks_()[b] = S.get_blocks_()[b].get({Accessor::range(0, kdim)});
        }
      }

      // remove:
      // vec_erase_(S.get_itoi(),to_be_removed);
      S.get_itoi() = new_itoi;
      vec_erase_(S.get_blocks_(), to_be_removed);
      // TODOfermionic: remove signs for this block
      //  vec_erase_(S.signflip(), to_be_removed);
      vec_erase_(S.bonds()[0].qnums(), to_be_removed);
      S.bonds()[0]._impl->_degs = new_dims;
      S.bonds()[0]._impl->_dim = tot_dim;
      S.bonds()[1] = S.bonds()[0].redirect();

      int t = 1;
      if (is_U) {
        UniTensor &U = outCyT[t];
        to_be_removed.clear();
        U.bonds().back() = S.bonds()[1].clone();
        std::vector<Accessor> acs(U.rank());
        for (int i = 0; i < U.rowrank(); i++) acs[i] = Accessor::all();

        for (int b = 0; b < U.Nblocks(); b++) {
          if (keep_dims[U.get_qindices(b).back()] == 0)
            to_be_removed.push_back(b);
          else {
            /// process blocks:
            if (keep_dims[U.get_qindices(b).back()] != U.get_blocks_()[b].shape().back()) {
              acs.back() = Accessor::range(0, keep_dims[U.get_qindices(b).back()]);
              U.get_blocks_()[b] = U.get_blocks_()[b].get(acs);
            }

            // change to new qindices:
            U.get_qindices(b).back() = new_qid[U.get_qindices(b).back()];
          }
        }
        vec_erase_(U.get_itoi(), to_be_removed);
        vec_erase_(U.get_blocks_(), to_be_removed);
        // TODOfermionic: remove signs for this block
        //  vec_erase_(U.signflip(), to_be_removed);
        t++;
      }

      if (is_vT) {
        UniTensor &vT = outCyT[t];
        to_be_removed.clear();
        vT.bonds().front() = S.bonds()[0].clone();
        std::vector<Accessor> acs(vT.rank());
        for (int i = 1; i < vT.rank(); i++) acs[i] = Accessor::all();

        for (int b = 0; b < vT.Nblocks(); b++) {
          if (keep_dims[vT.get_qindices(b)[0]] == 0)
            to_be_removed.push_back(b);
          else {
            /// process blocks:
            if (keep_dims[vT.get_qindices(b)[0]] != vT.get_blocks_()[b].shape()[0]) {
              acs[0] = Accessor::range(0, keep_dims[vT.get_qindices(b)[0]]);
              vT.get_blocks_()[b] = vT.get_blocks_()[b].get(acs);
            }
            // change to new qindices:
            vT.get_qindices(b)[0] = new_qid[vT.get_qindices(b)[0]];
          }
        }
        vec_erase_(vT.get_itoi(), to_be_removed);
        vec_erase_(vT.get_blocks_(), to_be_removed);
        // TODOfermionic: remove signs for this block
        //  vec_erase_(vT.signflip(), to_be_removed);
        t++;
      }

      // handle return_err!
      if (return_err == 1) {
        outCyT.push_back(UniTensor(Tensor({1}, Smin.dtype())));
        outCyT.back().get_block_().storage().at(0) = Smin;
      } else if (return_err) {
        outCyT.push_back(UniTensor(Sall.get({Accessor::tilend(smidx)})));
      }
    }  // _gesvd_truncate_BlockFermionic_UT

    std::vector<cytnx::UniTensor> Gesvd_truncate(const cytnx::UniTensor &Tin,
                                                 const cytnx_uint64 &keepdim, const double &err,
                                                 const bool &is_U, const bool &is_vT,
                                                 const unsigned int &return_err,
                                                 const cytnx_uint64 &mindim) {
      // using rowrank to split the bond to form a matrix.
      cytnx_error_msg(
        (Tin.rowrank() < 1 || Tin.rank() == 1 || Tin.rowrank() == Tin.rank()),
        "[ERROR][Gesvd_truncate] UniTensor should have rank>1 and rank>rowrank>0 for Svd%s", "\n");

      cytnx_error_msg(
        Tin.is_diag(),
        "[Gesvd_truncate][ERROR] SVD for diagonal UniTensor is trivial and currently not "
        "support. Use other manipulation.%s",
        "\n");

      // check input arguments
      cytnx_error_msg(mindim < 0, "[ERROR][Gesvd_truncate] mindim must be >=1%s", "\n");
      cytnx_error_msg(keepdim < 1, "[ERROR][Gesvd_truncate] keepdim must be >=1%s", "\n");
      cytnx_error_msg(return_err < 0, "[ERROR][Gesvd_truncate] return_err cannot be negative%s",
                      "\n");

      std::vector<UniTensor> outCyT;
      if (Tin.uten_type() == UTenType.Dense) {
        _gesvd_truncate_Dense_UT(outCyT, Tin, keepdim, err, is_U, is_vT, return_err, mindim);
      } else if (Tin.uten_type() == UTenType.Block) {
        _gesvd_truncate_Block_UT(outCyT, Tin, keepdim, err, is_U, is_vT, return_err, mindim);
      } else if (Tin.uten_type() == UTenType.BlockFermionic) {
        _gesvd_truncate_BlockFermionic_UT(outCyT, Tin, keepdim, err, is_U, is_vT, return_err,
                                          mindim);
      } else {
        cytnx_error_msg(
          true, "[ERROR] Gesvd_truncate only supports Dense/Block/BlockFermionic UniTensors.%s",
          "\n");
      }
      return outCyT;

    }  // Gesvd_truncate

    void _gesvd_truncate_Block_UT(std::vector<UniTensor> &outCyT, const cytnx::UniTensor &Tin,
                                  const cytnx_uint64 &keepdim,
                                  std::vector<cytnx_uint64> min_blockdim, const double &err,
                                  const bool &is_U, const bool &is_vT,
                                  const unsigned int &return_err, const cytnx_uint64 &mindim) {
      // currently, Gesvd is used as a standard for the full SVD before truncation
      cytnx_int64 keep_dim = keepdim;  // these must be signed int, because they can become
                                       // negative!
      cytnx_int64 min_dim = (mindim < 1 ? 1 : mindim);

      outCyT = linalg::Gesvd(Tin, is_U, is_vT);
      if (min_blockdim.size() == 1)  // if only one element given, make it a vector
        min_blockdim.resize(outCyT[0].Nblocks(), min_blockdim.front());
      cytnx_error_msg(
        min_blockdim.size() != outCyT[0].Nblocks(),
        "[ERROR][Gesvd_truncate] min_blockdim must have the same number of elements as "
        "blocks in the singular value UniTensor%s",
        "\n");

      // process truncation:
      // 1) concate all S vals from all blk but exclude the first min_blockdim Svals in each block
      // (since they will be kept anyways later)
      Tensor Sall;  // S vals excluding the already kept ones
      Tensor Block;  // current block
      cytnx_uint64 blockdim;
      bool anySall = false;  // are there already any values in Sall?
      bool any_min_blockdim = false;  // is any min_blockdim > 0?
      for (int b = 0; b < outCyT[0].Nblocks(); b++) {
        if (min_blockdim[b] < 1)  // save whole block to Sall
          Block = outCyT[0].get_block_(b);
        else {
          any_min_blockdim = true;
          blockdim = outCyT[0].get_block_(b).shape()[0];
          if (blockdim <= min_blockdim[b]) {
            // keep whole block
            keep_dim -= blockdim;
            min_dim -= blockdim;
            continue;
          }
          // remove first min_blockdim[b] values since they are saved anyways and do not need to be
          // included in Sall
          blockdim = outCyT[0].get_block_(b).shape()[0];
          Block = outCyT[0].get_block_(b).get({Accessor::range(min_blockdim[b], blockdim)});
          keep_dim -= min_blockdim[b];
          min_dim -= min_blockdim[b];
        }
        if (anySall)
          Sall = algo::Concatenate(Sall, Block);
        else {
          Sall = Block;
          anySall = true;
        }
      }
      if (!anySall) {
        // no truncation; return_err is tensor with one element, set to 0
        if (return_err >= 1) {
          outCyT.push_back(UniTensor(Tensor({1}, Tin.dtype())));
        }
      } else {
        Scalar Smin;
        if (keep_dim > 0) {
          if (!any_min_blockdim) {
            // make sure that at least one singular value is kept
            min_dim = (min_dim < 1 ? 1 : min_dim);
          } else {
            min_dim = (min_dim < 1 ? 0 : min_dim);
          }
          Sall = algo::Sort(Sall);  // all singular values, starting from the smallest
          // 2) get the minimum S value based on the args input.
          cytnx_uint64 smidx;
          cytnx_uint64 Sshape = Sall.shape()[0];
          if (keep_dim < Sshape) {
            smidx = Sshape - (cytnx_uint64)keep_dim;
            Smin = Sall.storage()(smidx);
          } else {
            keep_dim = Sshape;
            smidx = 0;
            Smin = Sall.storage()(0);
          }
          while ((Smin < err) and (keep_dim > min_dim)) {
            // at least one singular value is always kept!
            keep_dim--;
            if (keep_dim == 0) break;  // this is needed, keep_dim can be 0
            smidx++;
            Smin = Sall.storage()(smidx);
          }
          // handle return_err!
          if (return_err == 1) {
            outCyT.push_back(UniTensor(Tensor({1}, Smin.dtype())));
            outCyT.back().get_block_().storage().at(0) = Smin;
          } else if (return_err) {
            outCyT.push_back(UniTensor(Sall.get({Accessor::tilend(smidx)})));
          }
        } else {
          if (return_err >= 1) {
            outCyT.push_back(UniTensor(Tensor({1}, Tin.dtype())));
          }
        }

        // traversal each block and truncate!
        UniTensor &S = outCyT[0];
        std::vector<cytnx_uint64> new_dims;  // keep_dims for each block!
        std::vector<cytnx_int64> keep_dims;
        keep_dims.reserve(S.Nblocks());
        std::vector<cytnx_int64> new_qid;
        new_qid.reserve(S.Nblocks());

        std::vector<std::vector<cytnx_uint64>>
          new_itoi;  // assume S block is in same order as qnum:
        std::vector<cytnx_uint64> to_be_remove;

        cytnx_uint64 tot_dim = 0;
        cytnx_uint64 cnt = 0;
        for (int b = 0; b < S.Nblocks(); b++) {
          Storage stmp = S.get_block_(b).storage();
          cytnx_int64 kdim = min_blockdim[b];
          if (keep_dim > 0) {
            // search for first value >= Smin
            for (int i = stmp.size(); i > min_blockdim[b]; i--) {
              // Careful here: if (int i = stmp.size() -1; i >= min_blockdim[b]; i--) is used
              // instead, the compiler might make i an unsigned integer; if then min_blockdim[b] ==
              // 0, the condition i > min_blockdim[b] is always fulfilled and the loop never stops!
              if (stmp(i - 1) >= Smin) {
                kdim = i;
                break;
              }
            }
          }
          keep_dims.push_back(kdim);
          if (kdim == 0) {
            to_be_remove.push_back(b);
            new_qid.push_back(-1);
          } else {
            new_qid.push_back(new_dims.size());
            new_itoi.push_back({new_dims.size(), new_dims.size()});
            new_dims.push_back(kdim);
            tot_dim += kdim;
            if (kdim != S.get_blocks_()[b].shape()[0])
              S.get_blocks_()[b] = S.get_blocks_()[b].get({Accessor::range(0, kdim)});
          }
        }

        // remove:
        // vec_erase_(S.get_itoi(),to_be_remove);
        S.get_itoi() = new_itoi;
        vec_erase_(S.get_blocks_(), to_be_remove);
        vec_erase_(S.bonds()[0].qnums(), to_be_remove);
        S.bonds()[0]._impl->_degs = new_dims;
        S.bonds()[0]._impl->_dim = tot_dim;
        S.bonds()[1] = S.bonds()[0].redirect();

        int t = 1;
        if (is_U) {
          UniTensor &U = outCyT[t];
          to_be_remove.clear();
          U.bonds().back() = S.bonds()[1].clone();
          std::vector<Accessor> acs(U.rank());
          for (int i = 0; i < U.rowrank(); i++) acs[i] = Accessor::all();

          for (int b = 0; b < U.Nblocks(); b++) {
            if (keep_dims[U.get_qindices(b).back()] == 0)
              to_be_remove.push_back(b);
            else {
              /// process blocks:
              if (keep_dims[U.get_qindices(b).back()] != U.get_blocks_()[b].shape().back()) {
                acs.back() = Accessor::range(0, keep_dims[U.get_qindices(b).back()]);
                U.get_blocks_()[b] = U.get_blocks_()[b].get(acs);
              }

              // change to new qindices:
              U.get_qindices(b).back() = new_qid[U.get_qindices(b).back()];
            }
          }
          vec_erase_(U.get_itoi(), to_be_remove);
          vec_erase_(U.get_blocks_(), to_be_remove);

          t++;
        }

        if (is_vT) {
          UniTensor &vT = outCyT[t];
          to_be_remove.clear();
          vT.bonds().front() = S.bonds()[0].clone();
          std::vector<Accessor> acs(vT.rank());
          for (int i = 1; i < vT.rank(); i++) acs[i] = Accessor::all();

          for (int b = 0; b < vT.Nblocks(); b++) {
            if (keep_dims[vT.get_qindices(b)[0]] == 0)
              to_be_remove.push_back(b);
            else {
              /// process blocks:
              if (keep_dims[vT.get_qindices(b)[0]] != vT.get_blocks_()[b].shape()[0]) {
                acs[0] = Accessor::range(0, keep_dims[vT.get_qindices(b)[0]]);
                vT.get_blocks_()[b] = vT.get_blocks_()[b].get(acs);
              }
              // change to new qindices:
              vT.get_qindices(b)[0] = new_qid[vT.get_qindices(b)[0]];
            }
          }
          vec_erase_(vT.get_itoi(), to_be_remove);
          vec_erase_(vT.get_blocks_(), to_be_remove);
          t++;
        }
      }
    }  // _gesvd_truncate_Block_UT

    std::vector<cytnx::UniTensor> Gesvd_truncate(const cytnx::UniTensor &Tin,
                                                 const cytnx_uint64 &keepdim,
                                                 const std::vector<cytnx_uint64> min_blockdim,
                                                 const double &err, const bool &is_U,
                                                 const bool &is_vT, const unsigned int &return_err,
                                                 const cytnx_uint64 &mindim) {
      // using rowrank to split the bond to form a matrix.
      cytnx_error_msg(
        (Tin.rowrank() < 1 || Tin.rank() == 1 || Tin.rowrank() == Tin.rank()),
        "[ERROR][Gesvd_truncate] UniTensor should have rank>1 and rank>rowrank>0 for Svd%s", "\n");

      cytnx_error_msg(
        Tin.is_diag(),
        "[Gesvd_truncate][ERROR] SVD for diagonal UniTensor is trivial and currently not "
        "support. Use other manipulation.%s",
        "\n");

      // check input arguments
      // cytnx_error_msg(mindim < 0, "[ERROR][Gesvd_truncate] mindim must be >=1%s", "\n");
      cytnx_error_msg(keepdim < 1, "[ERROR][Gesvd_truncate] keepdim must be >=1%s", "\n");
      // cytnx_error_msg(return_err < 0, "[ERROR][Gesvd_truncate] return_err cannot be negative%s",
      //                 "\n");

      std::vector<UniTensor> outCyT;
      if (Tin.uten_type() == UTenType.Dense) {
        cytnx_error_msg(
          min_blockdim.size() != 1,
          "[ERROR][Gesvd_truncate] min_blockdim must have one element for dense UniTensor%s", "\n");
        _gesvd_truncate_Dense_UT(outCyT, Tin, keepdim, err, is_U, is_vT, return_err,
                                 max(mindim, min_blockdim[0]));

      } else if (Tin.uten_type() == UTenType.Block) {
        _gesvd_truncate_Block_UT(outCyT, Tin, keepdim, min_blockdim, err, is_U, is_vT, return_err,
                                 mindim);

      } else {
        cytnx_error_msg(
          true, "[ERROR][Gesvd_truncate] only Dense/Block UniTensors are supported.%s", "\n");
      }
      return outCyT;

    }  // Gesvd_truncate

  }  // namespace linalg
}  // namespace cytnx
#endif  // BACKEND_TORCH
