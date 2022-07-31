#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
#include <iostream>
#include <vector>

using namespace std;
namespace cytnx {
  namespace linalg {

    std::vector<Tensor> Svd(const Tensor &Tin, const bool &is_U, const bool &is_vT) {
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
      S.storage().set_zeros();
      if (is_U) {
        U.Init({in.shape()[0], n_singlu}, in.dtype(), in.device());
        U.storage().set_zeros();
      }
      if (is_vT) {
        vT.Init({n_singlu, in.shape()[1]}, in.dtype(), in.device());
        vT.storage().set_zeros();
      }

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Svd_ii[in.dtype()](
          in._impl->storage()._impl, U._impl->storage()._impl, vT._impl->storage()._impl,
          S._impl->storage()._impl, in.shape()[0], in.shape()[1]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_U) out.push_back(U);
        if (is_vT) out.push_back(vT);

        return out;

      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(in.device()));
        cytnx::linalg_internal::lii.cuSvd_ii[in.dtype()](
          in._impl->storage()._impl, U._impl->storage()._impl, vT._impl->storage()._impl,
          S._impl->storage()._impl, in.shape()[0], in.shape()[1]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_U) out.push_back(U);
        if (is_vT) out.push_back(vT);

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
    std::vector<cytnx::UniTensor> Svd(const cytnx::UniTensor &Tin, const bool &is_U,
                                      const bool &is_vT) {
      // using rowrank to split the bond to form a matrix.
      cytnx_error_msg(Tin.rowrank() < 1 || Tin.rank() == 1,
                      "[Svd][ERROR] Svd for DenseUniTensor should have rank>1 and rowrank>0%s",
                      "\n");

      if (Tin.uten_type() == UTenType.Sparse) {
        // cytnx_error_msg(true,"[Svd][Developing] Svd for SparseUniTensor is developing.%s","\n");

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

        for (int blk = 0; blk < comm_qnums.size(); blk++) {
          // std::cout << "QN block: " << blk << std::endl;
          int idd = 0;
          auto out = linalg::Svd(i_blocks[blk], is_U, is_vT);

          sls[blk] = out[idd];
          cytnx_uint64 deg = sls[blk].shape()[0];
          total_comm_dim += deg;

          std::vector<std::vector<cytnx_int64>> this_qnums(deg, comm_qnums[blk]);

          tmp_qns.insert(tmp_qns.end(), this_qnums.begin(), this_qnums.end());

          idd++;
          if (is_U) {
            Uls[blk] = out[idd];
            idd++;
          }
          if (is_vT) {
            vTls[blk] = out[idd];
          }
        }

        // std::cout << tmp_qns.size() << std::endl;
        // std::cout << total_comm_dim << std::endl;

        // construct common bond:
        Bond comm_bdi(total_comm_dim, bondType::BD_KET, tmp_qns);
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
        cytnx_error_msg(tmps->get_blocks_().size() != sls.size(), "[ERROR] internal error s.%s",
                        "\n");

        // wrapping:
        tmps->_blocks = sls;
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
          cytnx_error_msg(tmpu->get_blocks_().size() != Uls.size(), "[ERROR] internal error U.%s",
                          "\n");

          tmpu->_blocks = Uls;
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
          cytnx_error_msg(tmpv->get_blocks_().size() != vTls.size(), "[ERROR] internal error vT.%s",
                          "\n");

          tmpv->_blocks = vTls;
          UniTensor vT;
          vT._impl = boost::intrusive_ptr<UniTensor_base>(tmpv);
          outCyT.push_back(vT);
        }

        return outCyT;

      } else {
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
        vector<cytnx_int64> oldlabel = Tin.labels();

        // collapse as Matrix:
        cytnx_int64 rowdim = 1;
        for (cytnx_uint64 i = 0; i < Tin.rowrank(); i++) rowdim *= tmp.shape()[i];
        tmp.reshape_({rowdim, -1});

        vector<Tensor> outT = cytnx::linalg::Svd(tmp, is_U, is_vT);
        if (Tin.is_contiguous()) tmp.reshape_(oldshape);

        int t = 0;
        vector<cytnx::UniTensor> outCyT(outT.size());

        // s
        cytnx::UniTensor &Cy_S = outCyT[t];
        cytnx::Bond newBond(outT[t].shape()[0]);
        cytnx_int64 newlbl = -1;
        for (int i = 0; i < oldlabel.size(); i++) {
          if (oldlabel[i] <= newlbl) newlbl = oldlabel[i] - 1;
        }
        Cy_S.Init({newBond, newBond}, {newlbl, newlbl - 1}, 1, Type.Double, Device.cpu,
                  true);  // it is just reference so no hurt to alias ^^
        // cout << "[AFTER INIT]" << endl;
        Cy_S.put_block_(outT[t]);
        t++;
        if (is_U) {
          cytnx::UniTensor &Cy_U = outCyT[t];
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

        return outCyT;

      }  // is block form ?

    }  // Svd

  }  // namespace linalg
}  // namespace cytnx
