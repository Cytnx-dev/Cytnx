#include "linalg.hpp"
#include "Tensor.hpp"
#include <iostream>
#include <vector>
#include "UniTensor.hpp"
#include "algo.hpp"
using namespace std;
typedef cytnx::Accessor ac;

#ifdef BACKEND_TORCH
#else
  #include "../backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {

    std::vector<Tensor> Qdr(const Tensor &Tin, const bool &is_tau) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[Qdr] error, Qdr can only operate on rank-2 Tensor.%s", "\n");
      // cytnx_error_msg(!Tin.is_contiguous(), "[Qdr] error tensor must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      cytnx_uint64 n_tau = std::max(cytnx_uint64(1), std::min(Tin.shape()[0], Tin.shape()[1]));

      Tensor in = Tin.contiguous();
      if (Tin.dtype() > Type.Float) in = in.astype(Type.Double);

      // std::cout << n_singlu << std::endl;

      Tensor tau, Q, R, D;  // D is not used here.
      tau.Init({n_tau}, in.dtype(), in.device());
      tau.storage().set_zeros();
      Q.Init(Tin.shape(), in.dtype(), in.device());
      Q.storage().set_zeros();
      R.Init({n_tau, Tin.shape()[1]}, in.dtype(), in.device());
      R.storage().set_zeros();
      D = tau.clone();

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.QR_ii[in.dtype()](
          in._impl->storage()._impl, Q._impl->storage()._impl, R._impl->storage()._impl,
          D._impl->storage()._impl, tau._impl->storage()._impl, in.shape()[0], in.shape()[1], true);

        std::vector<Tensor> out;
        if (in.shape()[0] < in.shape()[1]) Q = Q[{ac::all(), ac::range(0, in.shape()[0], 1)}];
        out.push_back(Q);
        out.push_back(D);
        out.push_back(R);

        if (is_tau) out.push_back(tau);

        return out;

      } else {
  #ifdef UNI_GPU
        cytnx_error_msg(true, "[Qdr] error,%s", "Currently QR does not support CUDA.\n");
        /*
        checkCudaErrors(cudaSetDevice(in.device()));
        cytnx::linalg_internal::lii.cuQR_ii[in.dtype()](in._impl->storage()._impl,
                                                U._impl->storage()._impl,
                                                vT._impl->storage()._impl,
                                                S._impl->storage()._impl,in.shape()[0],in.shape()[1],
        false);

        std::vector<Tensor> out;
        out.push_back(S);
        if(is_U) out.push_back(U);
        if(is_vT) out.push_back(vT);

        return out;
        */
        return std::vector<Tensor>();
  #else
        cytnx_error_msg(true, "[QdR] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return std::vector<Tensor>();
  #endif
      }
    }

  }  // namespace linalg

}  // namespace cytnx

namespace cytnx {
  namespace linalg {
    std::vector<UniTensor> Qdr(const UniTensor &Tin, const bool &is_tau) {
      if (Tin.is_blockform()) {
        cytnx_error_msg(true, "[Qdr for sparse UniTensor is developling%s]", "\n");
      } else {
        // using rowrank to split the bond to form a matrix.
        cytnx_error_msg(Tin.rowrank() < 1 || Tin.rank() == 1,
                        "[Qdr][ERROR] Qdr for DenseUniTensor should have rank>1 and rowrank>0%s",
                        "\n");

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

        vector<Tensor> outT = cytnx::linalg::Qdr(tmp, is_tau);
        if (Tin.is_contiguous()) tmp.reshape_(oldshape);

        int t = 0;
        vector<cytnx::UniTensor> outCyT(outT.size());

        string newlbl = "_aux_L";

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

        // D
        outCyT[1] = UniTensor(outT[1], true, 1);
        // outCyT[1].set_labels({newlbl, newlbl - 1});
        // newlbl -= 1;
        outCyT[1].set_labels({newlbl, string("_aux_R")});
        newlbl = outCyT[1].labels().back();

        // R
        Qshape.clear();
        Qlbl.clear();
        Qshape.push_back(-1);
        Qlbl.push_back(newlbl);
        for (int i = Tin.rowrank(); i < Tin.rank(); i++) {
          Qshape.push_back(oldshape[i]);
          Qlbl.push_back(oldlabel[i]);
        }
        outT[2].reshape_(Qshape);
        outCyT[2] = UniTensor(outT[2], false, 1);
        outCyT[2].set_labels(Qlbl);

        // tau
        if (is_tau) {
          outCyT[3] = UniTensor(outT[3], false, 0);
        }

        if (Tin.is_tag()) {
          outCyT[0].tag();
          outCyT[1].tag();
          outCyT[2].tag();
          for (int i = 0; i < Tin.rowrank(); i++) {
            outCyT[0].bonds()[i].set_type(Tin.bonds()[i].type());
          }
          outCyT[0].bonds().back().set_type(cytnx::BD_BRA);
          outCyT[0]._impl->_is_braket_form = outCyT[0]._impl->_update_braket();

          outCyT[1].bonds()[0].set_type(cytnx::BD_KET);
          outCyT[1].bonds()[1].set_type(cytnx::BD_BRA);
          outCyT[1]._impl->_is_braket_form = outCyT[1]._impl->_update_braket();

          outCyT[2].bonds()[0].set_type(cytnx::BD_KET);
          for (int i = 1; i < outCyT[2].rank(); i++) {
            outCyT[2].bonds()[i].set_type(Tin.bonds()[Tin.rowrank() + i - 1].type());
          }
          outCyT[2]._impl->_is_braket_form = outCyT[2]._impl->_update_braket();
        }

        return outCyT;

      }  // block_form?
    }
  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
