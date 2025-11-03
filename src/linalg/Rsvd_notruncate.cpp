#include <string>
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
    std::vector<Tensor> Rsvd_notruncate(const cytnx::Tensor &Tin, cytnx_uint64 keepdim, bool is_U, bool is_vT,
                             cytnx_uint64 power_iteration, unsigned int seed) {
      std::vector<cytnx_uint64> shape = Tin.shape();
      cytnx_error_msg(shape.size() != 2, "[Rsvd] error, Rsvd can only operate on rank-2 Tensor.%s",
                      "\n");
      cytnx_error_msg(keepdim < 1, "[ERROR][Rsvd] Keepdim must be > 0, but is %d.\n", keepdim);

      Tensor in = Tin.contiguous();
      if (Tin.dtype() > Type.Float) in = in.astype(Type.Double);

      // form isometry Q[0] and apply Q.Dagger * in
      Tensor Q = Rand_isometry(Tin, keepdim, power_iteration, seed);
      in = Matmul(Q.Conj().permute_({1, 0}), in);

      shape = in.shape();
      cytnx_uint64 n_singlu = std::max(cytnx_uint64(1), std::min(shape[0], shape[1]));

      Tensor U, S, vT;
      S.Init({n_singlu}, in.dtype() <= 2 ? in.dtype() + 2 : in.dtype(),
             in.device());  // if type is complex, S should be real
      // S.storage().set_zeros();
      if (is_U) {
        U.Init({in.shape()[0], n_singlu}, in.dtype(), in.device());
        // U.storage().set_zeros();
      }
      if (is_vT) {
        vT.Init({n_singlu, in.shape()[1]}, in.dtype(), in.device());
        // vT.storage().set_zeros();
      }

      if (Tin.device() == Device.cpu) {
        // cytnx::linalg_internal::lii.Gesvd_ii[in.dtype()](
        //   in._impl->storage()._impl, U._impl->storage()._impl, vT._impl->storage()._impl,
        //   S._impl->storage()._impl, in.shape()[0], in.shape()[1]);
        cytnx::linalg_internal::lii.Gesvd_ii[in.dtype()](
          in._impl->storage()._impl, U._impl->storage()._impl, vT._impl->storage()._impl,
          S._impl->storage()._impl, in.shape()[0], in.shape()[1]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_U) {
          U = Matmul(Q, U);
          out.push_back(U);
        }
        if (is_vT) {
          out.push_back(vT);
        }

        return out;

      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(in.device()));
        cytnx::linalg_internal::lii.cuGeSvd_ii[in.dtype()](
          in._impl->storage()._impl, U._impl->storage()._impl, vT._impl->storage()._impl,
          S._impl->storage()._impl, in.shape()[0], in.shape()[1]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_U) {
          U = Matmul(Q, U);
          out.push_back(U);
        }
        if (is_vT) {
          out.push_back(vT);
        }

        return out;
  #else
        cytnx_error_msg(true, "[Rsvd] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return std::vector<Tensor>();
  #endif
      }
    }  // Rsvd(Tensor)

    void _Rsvd_notruncate_Dense_UT(std::vector<cytnx::UniTensor> &outCyT, const cytnx::UniTensor &Tin,
                        cytnx_uint64 keepdim, bool is_U, bool is_vT, cytnx_uint64 power_iteration,
                        unsigned int seed) {
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

      std::vector<cytnx_uint64> tmps = tmp.shape();
      std::vector<cytnx_int64> oldshape(tmps.begin(), tmps.end());
      tmps.clear();
      std::vector<std::string> oldlabel = Tin.labels();

      // collapse as Matrix:
      cytnx_int64 rowdim = 1;
      for (cytnx_uint64 i = 0; i < Tin.rowrank(); i++) rowdim *= tmp.shape()[i];
      tmp.reshape_({rowdim, -1});

      std::vector<Tensor> outT =
        cytnx::linalg::Rsvd_notruncate(tmp, keepdim, is_U, is_vT, power_iteration, seed);
      if (Tin.is_contiguous()) tmp.reshape_(oldshape);

      int t = 0;
      outCyT.resize(outT.size());

      // s
      cytnx::UniTensor &Cy_S = outCyT[t];
      cytnx::Bond newBond(outT[t].shape()[0]);

      Cy_S.Init({newBond, newBond}, {std::string("_aux_L"), std::string("_aux_R")}, 1, Type.Double,
                Tin.device(), true);  // it is just reference so no hurt to alias ^^

      // cout << "[AFTER INIT]" << endl;
      Cy_S.put_block_(outT[t]);
      t++;

      if (is_U) {
        cytnx::UniTensor &Cy_U = outCyT[t];
        cytnx_error_msg(Tin.rowrank() > oldshape.size(),
                        "[ERROR] The rowrank of the input unitensor is larger than the rank of the "
                        "contained tensor.%s",
                        "\n");
        std::vector<cytnx_int64> shapeU(oldshape.begin(), oldshape.begin() + Tin.rowrank());
        shapeU.push_back(-1);
        outT[t].reshape_(shapeU);
        Cy_U.Init(outT[t], false, Tin.rowrank());
        std::vector<std::string> labelU(oldlabel.begin(), oldlabel.begin() + Tin.rowrank());
        labelU.push_back(Cy_S.labels()[0]);
        Cy_U.set_labels(labelU);
        t++;  // U
      }
      if (is_vT) {
        cytnx::UniTensor &Cy_vT = outCyT[t];
        std::vector<cytnx_int64> shapevT(Tin.rank() - Tin.rowrank() + 1);
        shapevT[0] = -1;
        memcpy(&shapevT[1], &oldshape[Tin.rowrank()], sizeof(cytnx_int64) * (shapevT.size() - 1));

        outT[t].reshape_(shapevT);
        Cy_vT.Init(outT[t], false, 1);
        // cout << shapevT.size() << endl;
        std::vector<std::string> labelvT(shapevT.size());
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
    }  // _Rsvd_notruncate_Dense_UT

    std::vector<cytnx::UniTensor> Rsvd_notruncate(const cytnx::UniTensor &Tin, cytnx_uint64 keepdim, bool is_U,
                                       bool is_vT, cytnx_uint64 power_iteration,
                                       unsigned int seed) {
      // using rowrank to split the bond to form a matrix.
      cytnx_error_msg(Tin.rowrank() < 1 || Tin.rank() == 1,
                      "[Rsvd][ERROR] Rsvd for UniTensor should have rank>1 and rowrank>0%s", "\n");

      cytnx_error_msg(Tin.is_diag(),
                      "[Rsvd][ERROR] SVD for diagonal UniTensor is trivial and currently not "
                      "supported. Use other manipulations.%s",
                      "\n");

      std::vector<UniTensor> outCyT;
      if (Tin.uten_type() == UTenType.Dense) {
        _Rsvd_notruncate_Dense_UT(outCyT, Tin, keepdim, is_U, is_vT, power_iteration, seed);
        // } else if (Tin.uten_type() == UTenType.Block) {
        //   _Rsvd_notruncate_Block_UT(outCyT, Tin, keepdim, is_U, is_vT, power_iteration, seed);

        // } else if (Tin.uten_type() == UTenType.BlockFermionic) {
        //   _Rsvd_notruncate_BlockFermionic_UT(outCyT, Tin, keepdim, is_U, is_vT, power_iteration, seed);
      } else {
        cytnx_error_msg(true, "[ERROR] Rsvd currently only supports Dense UniTensors.%s", "\n");

      }  // is block form ?

      return outCyT;

    }  // Rsvd_notruncate
  }  // namespace linalg
}  // namespace cytnx
#endif  // BACKEND_TORCH
