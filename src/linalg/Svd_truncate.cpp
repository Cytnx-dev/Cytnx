#include "linalg.hpp"
#include "Accessor.hpp"
#include <vector>
#include "Tensor.hpp"
#include "UniTensor.hpp"
#include "algo.hpp"

namespace cytnx {
  namespace linalg {
    typedef Accessor ac;
    std::vector<Tensor> Svd_truncate(const Tensor &Tin, const cytnx_uint64 &keepdim,
                                     const double &err, const bool &is_U, const bool &is_vT,
                                     const bool &return_err) {
      std::vector<Tensor> tmps = Svd(Tin, is_U, is_vT);

      cytnx_uint64 id = 0;
      cytnx_uint64 Kdim = keepdim;

      Storage ts = tmps[0].storage();

      if (ts.size() < keepdim) {
        Kdim = ts.size();
      }

      cytnx_uint64 truc_dim = Kdim;
      for (cytnx_int64 i = Kdim - 1; i >= 0; i--) {
        if (ts.at(i) < err) {
          truc_dim--;
        } else {
          break;
        }
      }

      if (truc_dim == 0) {
        truc_dim = 1;
      }
      /// std::cout << truc_dim << std::endl;
      // cytnx_error_msg(tmps[0].shape()[0] < keepdim,"[ERROR] keepdim should be <= the valid # of
      // singular value, %d!\n",tmps[0].shape()[0]);
      Tensor terr({1}, Type.Double);

      if (truc_dim != ts.size()) {
        terr = tmps[id](truc_dim);
        tmps[id] = tmps[id].get({ac::range(0, truc_dim)});

        if (is_U) {
          id++;
          tmps[id] = tmps[id].get({ac::all(), ac::range(0, truc_dim)});
        }
        if (is_vT) {
          id++;
          tmps[id] = tmps[id].get({ac::range(0, truc_dim), ac::all()});
        }
      }
      if (return_err) tmps.push_back(terr);

      return tmps;
    }
  }  // namespace linalg
}  // namespace cytnx

namespace cytnx {
  namespace linalg {
    using namespace std;
    typedef Accessor ac;
    std::vector<cytnx::UniTensor> Svd_truncate(const cytnx::UniTensor &Tin,
                                               const cytnx_uint64 &keepdim, const double &err,
                                               const bool &is_U, const bool &is_vT,
                                               const bool &return_err) {
      // using rowrank to split the bond to form a matrix.
      cytnx_error_msg((Tin.rowrank() < 1 || Tin.rank() == 1),
                      "[Svd][ERROR] Svd for DenseUniTensor should have rank>1 and rowrank>0%s",
                      "\n");

      cytnx_uint64 keep_dim = keepdim;

      if (Tin.uten_type() == UTenType.Sparse) {
        // cytnx_error_msg(true,"[Svd][Developing] Svd for SparseUniTensor is developing.%s","\n");

        // Sparse UniTensor:

        UniTensor ipt = Tin.contiguous();

        cytnx_uint64 i_Rk = ipt.rank();
        cytnx_uint64 i_rowrank = ipt.rowrank();
        vector<Bond> Ubds;
        vector<Bond> vTbds(1);  // pre-set for left bd of vT
        auto comm_qnums = ipt.get_blocks_qnums();

        for (int i = 0; i < i_Rk; i++) {
          if (i < i_rowrank)
            Ubds.push_back(ipt.bonds()[i]);
          else
            vTbds.push_back(ipt.bonds()[i]);
        }

        // std::cout << Ubds << std::endl;
        // std::cout << vTbds << std::endl;

        // now, calculate svd for each blocks:
        std::vector<Tensor> Uls;
        std::vector<Tensor> sls(comm_qnums.size());
        std::vector<Tensor> vTls;

        if (is_U) Uls.resize(comm_qnums.size());
        if (is_vT) vTls.resize(comm_qnums.size());

        std::vector<Tensor> &i_blocks = ipt.get_blocks_();
        // std::vector<cytnx_uint64> degs(comm_qnums.size()); //deg of each blocks
        cytnx_uint64 total_comm_dim = 0;
        std::vector<std::vector<cytnx_int64>> tmp_qns;

        Tensor Sall;

        for (int blk = 0; blk < comm_qnums.size(); blk++) {
          // std::cout << "QN block: " << blk << std::endl;
          int idd = 0;
          auto out = linalg::Svd(i_blocks[blk], is_U, is_vT);

          sls[blk] = out[idd];
          if (Sall.dtype() == Type.Void)
            Sall = sls[blk];
          else
            Sall = algo::Concatenate(Sall, sls[blk]);

          // calculate new bond qnums:
          // cytnx_uint64 deg = sls[blk].shape()[0];
          // total_comm_dim+=deg;

          // std::vector< std::vector<cytnx_int64> > this_qnums(deg,comm_qnums[blk]);

          // tmp_qns.insert(tmp_qns.end(),this_qnums.begin(),this_qnums.end());
          //

          idd++;
          if (is_U) {
            Uls[blk] = out[idd];
            idd++;
          }
          if (is_vT) {
            vTls[blk] = out[idd];
          }
        }
        // cytnx_error_msg(keepdim>Sall.shape()[0],"[ERROR][Svd_truncate] keepdim should <=
        // dimension of total singular values%s","\n");
        std::vector<Tensor> o_sls, o_Uls, o_vTls;
        if (keepdim < Sall.shape()[0]) {  // keep_dim = Sall.shape()[0];

          // sorting:
          Sall = algo::Sort(Sall);  // small to large:
          // cout << Sall;
          // cout << Sall.shape()[0]-keepdim << endl;
          // cout << Sall(15);
          Scalar Smin = Sall(Sall.shape()[0] - keep_dim).item();

          std::vector<cytnx_int64> ambig_deg(comm_qnums.size());
          std::vector<cytnx_int64> degs(comm_qnums.size());
          // calculate new bond qnums and do truncate:
          for (int blk = 0; blk < comm_qnums.size(); blk++) {
            // std::cout << "QN block: " << blk << std::endl;
            int idd = 0;

            cytnx_int64 &deg = degs[blk];
            for (cytnx_int64 i = 0; i < sls[blk].shape()[0]; i++) {
              if (sls[blk](i).item() == Smin) {
                ambig_deg[blk]++;
              }
              if (sls[blk](i).item() >= Smin) {
                deg++;
              } else
                break;
            }
            total_comm_dim += deg;
          }

          // cout << degs << endl;
          // cout << total_comm_dim << endl;

          // checking

          // remove ambig_deg to fit keepdim:
          cytnx_int64 exceed = total_comm_dim - keep_dim;
          for (int blk = 0; blk < comm_qnums.size(); blk++) {
            if (exceed > 0) {
              if (ambig_deg[blk]) {
                if (ambig_deg[blk] > exceed) {
                  degs[blk] -= exceed;
                  exceed = 0;
                } else {
                  exceed -= ambig_deg[blk];
                  degs[blk] -= ambig_deg[blk];
                }
              }
            }

            // truncate
            if (degs[blk]) {
              std::vector<std::vector<cytnx_int64>> this_qnums(degs[blk], comm_qnums[blk]);
              tmp_qns.insert(tmp_qns.end(), this_qnums.begin(), this_qnums.end());

              // cout << "blk" << blk << "deg:" << degs[blk] << endl;

              // truncate:
              sls[blk] = sls[blk].get({ac::range(0, degs[blk])});
              o_sls.push_back(sls[blk]);

              if (is_U) {
                Uls[blk] = Uls[blk].get({ac::all(), ac::range(0, degs[blk])});
                if (Uls[blk].shape().size() == 1) Uls[blk].reshape_(Uls[blk].shape()[0], 1);
                o_Uls.push_back(Uls[blk]);
              }
              if (is_vT) {
                vTls[blk] = vTls[blk].get({ac::range(0, degs[blk]), ac::all()});
                if (vTls[blk].shape().size() == 1) vTls[blk].reshape_(1, vTls[blk].shape()[0]);
                o_vTls.push_back(vTls[blk]);
              }
            }
          }

        } else {
          keep_dim = Sall.shape()[0];
          for (int blk = 0; blk < comm_qnums.size(); blk++) {
            cytnx_uint64 deg = sls[blk].shape()[0];
            total_comm_dim += deg;

            std::vector<std::vector<cytnx_int64>> this_qnums(deg, comm_qnums[blk]);

            tmp_qns.insert(tmp_qns.end(), this_qnums.begin(), this_qnums.end());

            o_sls.push_back(sls[blk]);

            if (is_U) {
              o_Uls.push_back(Uls[blk]);
            }
            if (is_vT) {
              o_vTls.push_back(vTls[blk]);
            }
          }

        }  // if keepdim >= max dim

        // std::cout << tmp_qns.size() << std::endl;
        // std::cout << total_comm_dim << std::endl;

        // construct common bond:
        Bond comm_bdi(keep_dim, bondType::BD_KET, tmp_qns);
        Bond comm_bdo = comm_bdi.clone().set_type(bondType::BD_BRA);

        Ubds.push_back(comm_bdo);
        vTbds[0] = comm_bdi;

        // prepare output:
        std::vector<UniTensor> outCyT;

        vector<cytnx_int64> oldlabel = ipt.labels();
        cytnx_int64 newlbl = -1;
        for (int i = 0; i < oldlabel.size(); i++) {
          if (oldlabel[i] <= newlbl) newlbl = oldlabel[i] - 1;
        }

        // s
        SparseUniTensor *tmps = new SparseUniTensor();
        tmps->Init(
          {comm_bdi, comm_bdo}, {newlbl, newlbl - 1}, 1, Type.Double,
          Device.cpu, /* type and device does not matter here, cauz we are going to not alloc*/
          true, true);

        // check:
        cytnx_error_msg(tmps->get_blocks_().size() != o_sls.size(), "[ERROR] internal error s.%s",
                        "\n");

        // wrapping:
        tmps->_blocks = o_sls;
        UniTensor s;
        s._impl = boost::intrusive_ptr<UniTensor_base>(tmps);
        outCyT.push_back(s);

        if (is_U) {
          SparseUniTensor *tmpu = new SparseUniTensor();
          std::vector<cytnx_int64> LBLS = vec_clone(oldlabel, ipt.rowrank());
          LBLS.push_back(newlbl);
          tmpu->Init(
            Ubds, LBLS, ipt.rowrank(), Type.Double,
            Device.cpu, /* type and device does not matter here, cauz we are going to not alloc*/
            false, true);

          // check:
          cytnx_error_msg(tmpu->get_blocks_().size() != o_Uls.size(), "[ERROR] internal error U.%s",
                          "\n");

          tmpu->_blocks = o_Uls;

          UniTensor u;
          u._impl = boost::intrusive_ptr<UniTensor_base>(tmpu);
          outCyT.push_back(u);
        }

        if (is_vT) {
          SparseUniTensor *tmpv = new SparseUniTensor();
          std::vector<cytnx_int64> LBLS(ipt.rank() - ipt.rowrank() +
                                        1);  // old_label,ipt.rowrank());
          LBLS[0] = newlbl - 1;
          memcpy(&LBLS[1], &oldlabel[ipt.rowrank()],
                 sizeof(cytnx_int64) * (ipt.rank() - ipt.rowrank()));

          tmpv->Init(
            vTbds, LBLS, 1, Type.Double,
            Device.cpu, /* type and device does not matter here, cauz we are going to not alloc*/
            false, true);

          // check:
          cytnx_error_msg(tmpv->get_blocks_().size() != o_vTls.size(),
                          "[ERROR] internal error vT.%s", "\n");

          tmpv->_blocks = o_vTls;
          UniTensor vT;
          vT._impl = boost::intrusive_ptr<UniTensor_base>(tmpv);
          outCyT.push_back(vT);
        }

        return outCyT;

      } else {
        // DenseUniTensor:

        Tensor tmp = Tin.get_block_().contiguous();
        // if(Tin.is_contiguous()) tmp = Tin.get_block_();
        // else{ tmp = Tin.get_block(); tmp.contiguous_();}

        vector<cytnx_uint64> tmps = tmp.shape();
        vector<cytnx_int64> oldshape(tmps.begin(), tmps.end());
        tmps.clear();
        vector<cytnx_int64> oldlabel = Tin.labels();

        // collapse as Matrix:
        cytnx_int64 rowdim = 1;
        for (cytnx_uint64 i = 0; i < Tin.rowrank(); i++) rowdim *= tmp.shape()[i];
        tmp = tmp.reshape({rowdim, -1});

        vector<Tensor> outT =
          cytnx::linalg::Svd_truncate(tmp, keepdim, err, is_U, is_vT, return_err);

        // if(Tin.is_contiguous()) tmp.reshape_(oldshape);

        int t = 0;
        vector<cytnx::UniTensor> outCyT(outT.size());

        // s
        // cytnx_error_msg(keepdim>outT[t].shape()[0],"[ERROR][Svd_truncate] keepdim should <=
        // dimension of singular tensor%s","\n");

        cytnx::UniTensor &Cy_S = outCyT[t];
        cytnx::Bond newBond(outT[0].shape()[0]);
        cytnx_int64 newlbl = -1;
        for (int i = 0; i < oldlabel.size(); i++) {
          if (oldlabel[i] <= newlbl) newlbl = oldlabel[i] - 1;
        }
        Cy_S.Init({newBond, newBond}, {newlbl, newlbl - 1}, 1, Type.Double, Device.cpu,
                  true);  // it is just reference so no hurt to alias ^^
        Cy_S.put_block_(outT[t]);
        t++;

        if (is_U) {
          cytnx::UniTensor &Cy_U = outCyT[t];
          // shape
          vector<cytnx_int64> shapeU = vec_clone(oldshape, Tin.rowrank());
          shapeU.push_back(-1);

          outT[t].reshape_(shapeU);

          Cy_U.Init(outT[t], false, Tin.rowrank());
          vector<cytnx_int64> labelU = vec_clone(oldlabel, Tin.rowrank());
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
          vector<cytnx_int64> labelvT(shapevT.size());
          labelvT[0] = Cy_S.labels()[1];
          memcpy(&labelvT[1], &oldlabel[Tin.rowrank()], sizeof(cytnx_int64) * (labelvT.size() - 1));
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

        return outCyT;

      }  // is block form ?

    }  // Svd_truncate

  }  // namespace linalg
}  // namespace cytnx