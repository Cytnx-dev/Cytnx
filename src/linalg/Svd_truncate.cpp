#include "linalg.hpp"

#include <iostream>
#include <string>
#include <vector>

#include "Tensor.hpp"
#include "UniTensor.hpp"
#include "algo.hpp"
#include "Accessor.hpp"
#include "block_truncation_helpers.hpp"

#ifdef BACKEND_TORCH
#else

  #include "backend/linalg_internal_interface.hpp"
namespace cytnx {
  namespace linalg {

    std::vector<Tensor> Svd_truncate(const Tensor &Tin, const cytnx_uint64 &keepdim,
                                     const double &err, const bool &is_UvT,
                                     const unsigned int &return_err, const cytnx_uint64 &mindim) {
      // check input arguments
      cytnx_error_msg(keepdim < 1, "[ERROR][Svd_truncate] keepdim must be >=1.%s", "\n");
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[ERROR][Svd_truncate] can only operate on rank-2 Tensor.%s", "\n");

      if (Tin.device() == Device.cpu) {
        std::vector<Tensor> outT = Svd(Tin, is_UvT);

        // Svd packs is_UvT as both U and vT, so pass it for both flags. memcpyTruncation truncates
        // outT in place and appends the error tensor when return_err != 0.
        cytnx::linalg_internal::memcpyTruncation(outT, keepdim, err, is_UvT, is_UvT, return_err,
                                                 mindim);

        return outT;
      } else {
  #ifdef UNI_GPU
        std::vector<Tensor> outT = Svd(Tin, is_UvT);
        cytnx::linalg_internal::cudaMemcpyTruncation(outT, keepdim, err, is_UvT, is_UvT, return_err,
                                                     mindim);
        return outT;
  #else
        cytnx_error_msg(
          true, "[Error][Svd_truncate] Trying to call the gpu section without CUDA support%s",
          "\n");
        return std::vector<Tensor>();
  #endif
      }
    }

    static void Svd_truncate_Dense_UT_internal(std::vector<UniTensor> &outCyT,
                                               const cytnx::UniTensor &Tin,
                                               const cytnx_uint64 &keepdim, const double &err,
                                               const bool &is_UvT, const unsigned int &return_err,
                                               const cytnx_uint64 &mindim) {
      // DenseUniTensor:
      cytnx_uint64 keep_dim = keepdim;

      Tensor tmp = Tin.get_block_().contiguous();

      std::vector<cytnx_uint64> tmps = tmp.shape();
      std::vector<cytnx_int64> oldshape(tmps.begin(), tmps.end());
      tmps.clear();
      std::vector<std::string> oldlabel = Tin.labels();

      // collapse as Matrix:
      cytnx_int64 rowdim = 1;
      for (cytnx_uint64 i = 0; i < Tin.rowrank(); i++) rowdim *= tmp.shape()[i];
      tmp = tmp.reshape({rowdim, -1});

      std::vector<Tensor> outT =
        cytnx::linalg::Svd_truncate(tmp, keepdim, err, is_UvT, return_err, mindim);

      int t = 0;
      outCyT.resize(outT.size());

      // s

      cytnx::UniTensor &Cy_S = outCyT[t];
      cytnx::Bond newBond(outT[0].shape()[0]);
      Cy_S.Init({newBond, newBond}, {std::string("_aux_L"), std::string("_aux_R")},
                1,  // rowrank
                outT[t].dtype(), outT[t].device(),  // match the block that is inserted below
                true);  // is_diag
      Cy_S.put_block_(outT[t]);
      t++;

      if (is_UvT) {
        cytnx::UniTensor &Cy_U = outCyT[t];
        // shape
        cytnx_error_msg(Tin.rowrank() > oldshape.size(),
                        "[ERROR] The rowrank of the input UniTensor is larger than the rank of the "
                        "contained tensor.%s",
                        "\n");
        std::vector<cytnx_int64> shapeU(oldshape.begin(), oldshape.begin() + Tin.rowrank());
        shapeU.push_back(-1);

        outT[t].reshape_(shapeU);

        Cy_U.Init(outT[t], false, Tin.rowrank());
        std::vector<std::string> labelU(oldlabel.begin(), oldlabel.begin() + Tin.rowrank());
        labelU.push_back(Cy_S.labels()[0]);
        Cy_U.relabel_(labelU);
        t++;  // U
      }

      if (is_UvT) {
        cytnx::UniTensor &Cy_vT = outCyT[t];

        // shape
        std::vector<cytnx_int64> shapevT(Tin.rank() - Tin.rowrank() + 1);
        shapevT[0] = -1;
        memcpy(&shapevT[1], &oldshape[Tin.rowrank()], sizeof(cytnx_int64) * (shapevT.size() - 1));

        outT[t].reshape_(shapevT);

        Cy_vT.Init(outT[t], false, 1);
        std::vector<std::string> labelvT(shapevT.size());
        labelvT[0] = Cy_S.labels()[1];
        std::copy(oldlabel.begin() + Tin.rowrank(), oldlabel.end(), labelvT.begin() + 1);
        Cy_vT.relabel_(labelvT);
        t++;  // vT
      }

      // if tag, then update  the tagging informations
      if (Tin.is_tag()) {
        Cy_S.tag_();
        t = 1;
        if (is_UvT) {
          cytnx::UniTensor &Cy_U = outCyT[t];
          Cy_U._impl->_is_tag = true;
          for (int i = 0; i < Cy_U.rowrank(); i++) {
            Cy_U._impl->_bonds[i] = Cy_U._impl->_bonds[i].retype(Tin.bonds()[i].type());
          }
          Cy_U._impl->_bonds.back() = Cy_U._impl->_bonds.back().retype(cytnx::BD_BRA);
          Cy_U._impl->_is_braket_form = Cy_U._impl->_update_braket();
          t++;
        }
        if (is_UvT) {
          cytnx::UniTensor &Cy_vT = outCyT[t];
          Cy_vT._impl->_is_tag = true;
          Cy_vT._impl->_bonds[0] = Cy_vT._impl->_bonds[0].retype(cytnx::BD_KET);
          for (int i = 1; i < Cy_vT.rank(); i++) {
            Cy_vT._impl->_bonds[i] =
              Cy_vT._impl->_bonds[i].retype(Tin.bonds()[Tin.rowrank() + i - 1].type());
          }
          Cy_vT._impl->_is_braket_form = Cy_vT._impl->_update_braket();
          t++;
        }

      }  // if tag

      if (return_err) outCyT.back().Init(outT.back(), false, 0);
    }  // Svd_truncate_Dense_UT_internal

    static void Svd_truncate_Block_UTs_internal(std::vector<UniTensor> &outCyT,
                                                const cytnx::UniTensor &Tin,
                                                const cytnx_uint64 &keepdim, const double &err,
                                                const bool &is_UvT, const int &return_err,
                                                const cytnx_uint64 &mindim) {
      // currently, Gesvd is used as a standard for the full SVD before truncation
      // handles BlockFermionicUniTensor as well: elements of _signflip are removed if blocks are
      // erased
      cytnx_uint64 keep_dim = keepdim;

      outCyT = linalg::Gesvd(Tin, is_UvT, is_UvT);

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
      // the per-block scans below keep every value >= Smin, so an exact degeneracy at the
      // cut is kept entirely; only values strictly below Smin are dropped.
      smidx = CountDroppedSingularValues(Sall, smidx, Smin);

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
      S.get_itoi() = new_itoi;
      if (!to_be_removed.empty()) {
        vec_erase_(S.get_blocks_(), to_be_removed);
        vec_erase_(S._impl->_bonds[0]._impl->_qnums, to_be_removed);
        if (Tin.uten_type() == UTenType.BlockFermionic) {
          static_cast<BlockFermionicUniTensor *>(S._impl.get())->erase_signflip_(to_be_removed);
        }
      }
      S._impl->_bonds[0]._impl->_degs = new_dims;
      S._impl->_bonds[0]._impl->_dim = tot_dim;
      S._impl->_bonds[1] = S._impl->_bonds[0].redirect();

      int t = 1;
      if (is_UvT) {
        // depends on S.bonds()[1], keep_dims, new_qid
        UniTensor &U = outCyT[t];
        to_be_removed.clear();
        U._impl->_bonds.back() = S._impl->_bonds[1].clone();
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
        if (!to_be_removed.empty()) {
          vec_erase_(U.get_itoi(), to_be_removed);
          vec_erase_(U.get_blocks_(), to_be_removed);
          if (Tin.uten_type() == UTenType.BlockFermionic) {
            static_cast<BlockFermionicUniTensor *>(U._impl.get())->erase_signflip_(to_be_removed);
          }
        }

        t++;
      }

      if (is_UvT) {
        UniTensor &vT = outCyT[t];
        to_be_removed.clear();
        vT._impl->_bonds.front() = S._impl->_bonds[0].clone();
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
        if (!to_be_removed.empty()) {
          vec_erase_(vT.get_itoi(), to_be_removed);
          vec_erase_(vT.get_blocks_(), to_be_removed);
          if (Tin.uten_type() == UTenType.BlockFermionic) {
            static_cast<BlockFermionicUniTensor *>(vT._impl.get())->erase_signflip_(to_be_removed);
          }
        }

        t++;
      }

      // handle return_err!
      if (return_err) {
        outCyT.push_back(BuildBlockDiscardedSingularValues(Sall, smidx, return_err));
      }
    }  // Svd_truncate_Block_UTs_internal

    std::vector<cytnx::UniTensor> Svd_truncate(const cytnx::UniTensor &Tin,
                                               const cytnx_uint64 &keepdim, const double &err,
                                               const bool &is_UvT, const unsigned int &return_err,
                                               const cytnx_uint64 &mindim) {
      // using rowrank to split the bond to form a matrix.
      cytnx_error_msg(Tin.rank() <= 1,
                      "[ERROR][Svd_truncate] Input UniTensor should have rank>1, but rank is %d\n",
                      Tin.rank());
      cytnx_error_msg(
        Tin.rowrank() < 1,
        "[ERROR][Svd_truncate] Input UniTensor should have rowrank>0, but rowrank is %d\n",
        Tin.rowrank());
      cytnx_error_msg(Tin.rowrank() >= Tin.rank(),
                      "[ERROR][Svd_truncate] Input UniTensor should have rowrank<rank, but rowrank "
                      "is %d and rank is %d\n",
                      Tin.rowrank(), Tin.rank());
      cytnx_error_msg(Tin.is_diag(),
                      "[ERROR][Svd_truncate] Input UniTensor is diagonal, so Svd_truncate is "
                      "trivial and not supported. Use other manipulation.%s",
                      "\n");

      // check input arguments
      cytnx_error_msg(keepdim < 1, "[ERROR][Svd_truncate] keepdim must be >=1%s", "\n");

      std::vector<UniTensor> outCyT;
      if (Tin.uten_type() == UTenType.Dense) {
        Svd_truncate_Dense_UT_internal(outCyT, Tin, keepdim, err, is_UvT, return_err, mindim);

      } else if ((Tin.uten_type() == UTenType.Block) ||
                 (Tin.uten_type() == UTenType.BlockFermionic)) {
        Svd_truncate_Block_UTs_internal(outCyT, Tin, keepdim, err, is_UvT, return_err, mindim);
      } else {
        cytnx_error_msg(true, "[ERROR][Svd_truncate] UniTensor type '%s' not supported\n",
                        Tin.uten_type_str().c_str());
      }
      return outCyT;

    }  // Svd_truncate

    static void Svd_truncate_Block_UTs_internal(std::vector<UniTensor> &outCyT,
                                                const cytnx::UniTensor &Tin,
                                                const cytnx_uint64 &keepdim,
                                                std::vector<cytnx_uint64> min_blockdim,
                                                const double &err, const bool &is_UvT,
                                                const int &return_err, const cytnx_uint64 &mindim) {
      // currently, Gesvd is used as a standard for the full SVD before truncation
      // handles BlockFermionicUniTensor as well: elements of _signflip are removed if blocks are
      // erased
      cytnx_int64 keep_dim = keepdim;  // these must be signed int, because they can become
                                       // negative!
      cytnx_int64 min_dim = (mindim < 1 ? 1 : mindim);
      cytnx_error_msg(min_blockdim.empty(),
                      "[ERROR][Svd_truncate] min_blockdim must not be empty; use the overload "
                      "without min_blockdim if no per-block floor is needed.%s",
                      "\n");

      outCyT = linalg::Gesvd(Tin, is_UvT, is_UvT);
      if (min_blockdim.size() == 1)  // if only one element given, make it a vector
        min_blockdim.resize(outCyT[0].Nblocks(), min_blockdim.front());
      cytnx_error_msg(min_blockdim.size() != outCyT[0].Nblocks(),
                      "[ERROR][Svd_truncate] min_blockdim must have the same number of elements as "
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
        if (return_err >= 1) {
          outCyT.push_back(
            BuildNoDiscardedSingularValues(outCyT[0].dtype(), return_err, outCyT[0].device()));
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
          if (keep_dim == 0) {
            // the err threshold dropped every value in Sall; nothing above the cut is kept
            smidx = Sshape;
          } else {
            // the per-block scans below keep every value >= Smin, so an exact degeneracy at
            // the cut is kept entirely; only values strictly below Smin are dropped.
            smidx = CountDroppedSingularValues(Sall, smidx, Smin);
          }
          // handle return_err!
          if (return_err) {
            outCyT.push_back(BuildBlockDiscardedSingularValues(Sall, smidx, return_err));
          }
        } else {
          // keep_dim < 1: per-block min_blockdim guarantees already cover the global cap, so
          // every value in Sall is dropped.
          if (return_err) {
            Sall = algo::Sort(Sall);  // ascending; BuildBlockDiscardedSingularValues expects this
            outCyT.push_back(BuildBlockDiscardedSingularValues(Sall, Sall.shape()[0], return_err));
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
        std::vector<cytnx_uint64> to_be_removed;

        cytnx_uint64 tot_dim = 0;
        cytnx_uint64 cnt = 0;
        for (int b = 0; b < S.Nblocks(); b++) {
          Storage stmp = S.get_block_(b).storage();
          cytnx_int64 kdim = std::min<cytnx_int64>(min_blockdim[b], stmp.size());
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
        S.get_itoi() = new_itoi;
        if (!to_be_removed.empty()) {
          vec_erase_(S.get_blocks_(), to_be_removed);
          vec_erase_(S._impl->_bonds[0]._impl->_qnums, to_be_removed);
          if (Tin.uten_type() == UTenType.BlockFermionic) {
            static_cast<BlockFermionicUniTensor *>(S._impl.get())->erase_signflip_(to_be_removed);
          }
        }
        S._impl->_bonds[0]._impl->_degs = new_dims;
        S._impl->_bonds[0]._impl->_dim = tot_dim;
        S._impl->_bonds[1] = S._impl->_bonds[0].redirect();

        int t = 1;
        if (is_UvT) {
          UniTensor &U = outCyT[t];
          to_be_removed.clear();
          U._impl->_bonds.back() = S._impl->_bonds[1].clone();
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
          if (!to_be_removed.empty()) {
            vec_erase_(U.get_itoi(), to_be_removed);
            vec_erase_(U.get_blocks_(), to_be_removed);
            if (Tin.uten_type() == UTenType.BlockFermionic) {
              static_cast<BlockFermionicUniTensor *>(U._impl.get())->erase_signflip_(to_be_removed);
            }
          }

          t++;
        }

        if (is_UvT) {
          UniTensor &vT = outCyT[t];
          to_be_removed.clear();
          vT._impl->_bonds.front() = S._impl->_bonds[0].clone();
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
          if (!to_be_removed.empty()) {
            vec_erase_(vT.get_itoi(), to_be_removed);
            vec_erase_(vT.get_blocks_(), to_be_removed);
            if (Tin.uten_type() == UTenType.BlockFermionic) {
              static_cast<BlockFermionicUniTensor *>(vT._impl.get())
                ->erase_signflip_(to_be_removed);
            }
          }

          t++;
        }
      }
    }  // Svd_truncate_Block_UTs_internal

    std::vector<cytnx::UniTensor> Svd_truncate(const cytnx::UniTensor &Tin,
                                               const cytnx_uint64 &keepdim,
                                               const std::vector<cytnx_uint64> min_blockdim,
                                               const double &err, const bool &is_UvT,
                                               const unsigned int &return_err,
                                               const cytnx_uint64 &mindim) {
      // using rowrank to split the bond to form a matrix.
      cytnx_error_msg(Tin.rank() <= 1,
                      "[ERROR][Svd_truncate] Input UniTensor should have rank>1, but rank is %d\n",
                      Tin.rank());
      cytnx_error_msg(
        Tin.rowrank() < 1,
        "[ERROR][Svd_truncate] Input UniTensor should have rowrank>0, but rowrank is %d\n",
        Tin.rowrank());
      cytnx_error_msg(Tin.rowrank() >= Tin.rank(),
                      "[ERROR][Svd_truncate] Input UniTensor should have rowrank<rank, but rowrank "
                      "is %d and rank is %d\n",
                      Tin.rowrank(), Tin.rank());
      cytnx_error_msg(Tin.is_diag(),
                      "[ERROR][Svd_truncate] Input UniTensor is diagonal, so Svd_truncate is "
                      "trivial and not supported. Use other manipulation.%s",
                      "\n");

      // check input arguments
      cytnx_error_msg(keepdim < 1, "[ERROR][Svd_truncate] keepdim must be >=1%s", "\n");

      std::vector<UniTensor> outCyT;
      if (Tin.uten_type() == UTenType.Dense) {
        cytnx_error_msg(
          min_blockdim.size() != 1,
          "[ERROR][Svd_truncate] min_blockdim must have one element for dense UniTensor%s", "\n");
        Svd_truncate_Dense_UT_internal(outCyT, Tin, keepdim, err, is_UvT, return_err,
                                       std::max(mindim, min_blockdim[0]));

      } else if ((Tin.uten_type() == UTenType.Block) ||
                 (Tin.uten_type() == UTenType.BlockFermionic)) {
        Svd_truncate_Block_UTs_internal(outCyT, Tin, keepdim, min_blockdim, err, is_UvT, return_err,
                                        mindim);
      } else {
        cytnx_error_msg(true, "[ERROR][Svd_truncate] UniTensor type '%s' not supported\n",
                        Tin.uten_type_str().c_str());
      }
      return outCyT;

    }  // Svd_truncate

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
