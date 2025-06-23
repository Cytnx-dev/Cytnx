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
    typedef Accessor ac;
    std::vector<Tensor> Rsvd_truncate(const Tensor &Tin, const cytnx_uint64 &keepdim,
                                      const double &err, const bool &is_U, const bool &is_vT,
                                      const unsigned int &return_err, const cytnx_uint64 &mindim,
                                      const cytnx_uint64 &oversampling_summand,
                                      const double &oversampling_factor,
                                      const cytnx_uint64 &power_iteration,
                                      const unsigned int &seed) {
      // check input arguments
      cytnx_error_msg(mindim < 0, "[ERROR][Rsvd_truncate] mindim must be >=1.%s", "\n");
      cytnx_error_msg(keepdim < 1, "[ERROR][Rsvd_truncate] keepdim must be >=1.%s", "\n");
      cytnx_error_msg(return_err < 0, "[ERROR][Rsvd_truncate] return_err cannot be negative%s",
                      "\n");
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[Rsvd_truncate] can only operate on rank-2 Tensor.%s", "\n");
      cytnx_uint64 samplenum =
        (cytnx_uint64)((std::max(0., oversampling_factor) + 1.) * (double)keepdim) +
        oversampling_summand;
      cytnx_uint64 n_singlu = std::max(cytnx_uint64(1), std::min(Tin.shape()[0], Tin.shape()[1]));
      Tensor Q;
      if (Tin.device() == Device.cpu) {
        std::vector<Tensor> tmps;
        if (samplenum < n_singlu) {
          Tensor in = Tin.contiguous();
          Q = linalg::Rand_isometry(in, samplenum, power_iteration, seed);
          tmps = Gesvd(Matmul(Q.Conj().permute_({1, 0}), in), is_U, is_vT);  // run full SVD
        } else {
          tmps = Gesvd(Tin, is_U, is_vT);  // run full SVD
        }
        Tensor terr({1}, Tin.dtype(), Tin.device());

        cytnx::linalg_internal::lii.memcpyTruncation_ii[Tin.dtype()](
          tmps[1], tmps[2], tmps[0], terr, keepdim, err, is_U, is_vT, return_err, mindim);

        std::vector<Tensor> outT;
        outT.push_back(tmps[0]);
        if (is_U) {
          if (samplenum < n_singlu)
            outT.push_back(Matmul(Q, tmps[1]));
          else
            outT.push_back(tmps[1]);
        }
        if (is_vT) outT.push_back(tmps[2]);
        if (return_err) outT.push_back(terr);

        return outT;

      } else {
  #ifdef UNI_GPU
    #ifdef UNI_CUQUANTUM
        Tensor in = Tin.contiguous();
        // if (Tin.dtype() > Type.Float) in = in.astype(Type.Double);
        // prepare U, S, vT
        Tensor U, S, vT, terr;
        S.Init({n_singlu}, in.dtype() <= 2 ? in.dtype() + 2 : in.dtype(),
               in.device());  // if type is complex, S should be real
        U.Init({in.shape()[0], n_singlu}, in.dtype(), in.device());
        vT.Init({n_singlu, in.shape()[1]}, in.dtype(), in.device());
        terr.Init({1}, in.dtype(), in.device());
        if (samplenum < n_singlu) {
          Q = linalg::Rand_isometry(Tin, samplenum, power_iteration, seed);
          in = Matmul(Q.Conj().permute_({1, 0}), in)
        }
        cytnx::linalg_internal::lii.cuQuantumGeSvd_ii[in.dtype()](in, keepdim, err, return_err, U,
                                                                  S, vT, terr);

        cytnx::linalg_internal::lii.cudaMemcpyTruncation_ii[in.dtype()](
          U, vT, S, terr, keepdim, err, is_U, is_vT, return_err, mindim);

        std::vector<Tensor> outT;
        outT.push_back(S);
        if (is_U) {
          if (samplenum < n_singlu)
            outT.push_back(Matmul(Q, U));
          else
            outT.push_back(U);
        }
        if (is_vT) outT.push_back(vT);
        if (return_err) outT.push_back(terr);

        return outT;

    #else
        std::vector<Tensor> tmps;
        if (samplenum < n_singlu) {
          Tensor in = Tin.contiguous();
          Q = linalg::Rand_isometry(in, samplenum, power_iteration, seed);
          tmps = Gesvd(Matmul(Q.Conj().permute_({1, 0}), in), is_U, is_vT);  // run full SVD
        } else {
          tmps = Gesvd(Tin, is_U, is_vT);  // run full SVD
        }
        Tensor terr({1}, Tin.dtype(), Tin.device());

        cytnx::linalg_internal::lii.cudaMemcpyTruncation_ii[Tin.dtype()](
          tmps[1], tmps[2], tmps[0], terr, keepdim, err, is_U, is_vT, return_err, mindim);

        std::vector<Tensor> outT;
        outT.push_back(tmps[0]);
        if (is_U) {
          if (samplenum < n_singlu)
            outT.push_back(Matmul(Q, tmps[1]));
          else
            outT.push_back(tmps[1]);
        }
        if (is_vT) outT.push_back(tmps[2]);
        if (return_err) outT.push_back(terr);

        return outT;
    #endif
  #else
        cytnx_error_msg(
          true, "[Error][Rsvd_truncate] Trying to call the gpu section without CUDA support%s",
          "\n");
        return std::vector<Tensor>();
  #endif
      }
    }  // Rsvd_truncate(Tensor)

    void _Rsvd_truncate_Dense_UT(std::vector<UniTensor> &outCyT, const cytnx::UniTensor &Tin,
                                 const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
                                 const bool &is_vT, const unsigned int &return_err,
                                 const cytnx_uint64 &mindim,
                                 const cytnx_uint64 &oversampling_summand,
                                 const double &oversampling_factor,
                                 const cytnx_uint64 &power_iteration, const unsigned int &seed) {
      // DenseUniTensor:
      cytnx_uint64 keep_dim = keepdim;

      Tensor tmp = Tin.get_block_().contiguous();
      // if(Tin.is_contiguous()) tmp = Tin.get_block_();
      // else{ tmp = Tin.get_block(); tmp.contiguous_();}

      std::vector<cytnx_uint64> tmps = tmp.shape();
      std::vector<cytnx_int64> oldshape(tmps.begin(), tmps.end());
      tmps.clear();
      std::vector<std::string> oldlabel = Tin.labels();

      // collapse as Matrix:
      cytnx_int64 rowdim = 1;
      for (cytnx_uint64 i = 0; i < Tin.rowrank(); i++) rowdim *= tmp.shape()[i];
      tmp = tmp.reshape({rowdim, -1});

      std::vector<Tensor> outT = cytnx::linalg::Rsvd_truncate(
        tmp, keepdim, err, is_U, is_vT, return_err, mindim, oversampling_summand,
        oversampling_factor, power_iteration, seed);

      // if(Tin.is_contiguous()) tmp.reshape_(oldshape);

      int t = 0;
      outCyT.resize(outT.size());

      // s
      // cytnx_error_msg(keepdim>outT[t].shape()[0],"[ERROR][Rsvd_truncate] keepdim should <=
      // dimension of singular tensor%s","\n");

      cytnx::UniTensor &Cy_S = outCyT[t];
      cytnx::Bond newBond(outT[0].shape()[0]);
      Cy_S.Init({newBond, newBond}, {std::string("_aux_L"), std::string("_aux_R")}, 1, Type.Double,
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

        // shape
        std::vector<cytnx_int64> shapevT(Tin.rank() - Tin.rowrank() + 1);
        shapevT[0] = -1;
        memcpy(&shapevT[1], &oldshape[Tin.rowrank()], sizeof(cytnx_int64) * (shapevT.size() - 1));

        outT[t].reshape_(shapevT);

        Cy_vT.Init(outT[t], false, 1);
        std::vector<std::string> labelvT(shapevT.size());
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
    };  // _Rsvd_truncate_Dense_UT

    std::vector<cytnx::UniTensor> Rsvd_truncate(
      const cytnx::UniTensor &Tin, const cytnx_uint64 &keepdim, const double &err, const bool &is_U,
      const bool &is_vT, const unsigned int &return_err, const cytnx_uint64 &mindim,
      const cytnx_uint64 &oversampling_summand, const double &oversampling_factor,
      const cytnx_uint64 &power_iteration, const unsigned int &seed) {
      // using rowrank to split the bond to form a matrix.
      cytnx_error_msg(
        (Tin.rowrank() < 1 || Tin.rank() == 1 || Tin.rowrank() == Tin.rank()),
        "[ERROR][Rsvd_truncate] UniTensor should have rank>1 and rank>rowrank>0 for Svd%s", "\n");

      // check input arguments
      // cytnx_error_msg(mindim < 0, "[ERROR][Rsvd_truncate] mindim must be >=1%s", "\n");
      cytnx_error_msg(keepdim < 1, "[ERROR][Rsvd_truncate] keepdim must be >=1%s", "\n");
      // cytnx_error_msg(return_err < 0, "[ERROR][Rsvd_truncate] return_err cannot be negative%s",
      //                 "\n");

      std::vector<UniTensor> outCyT;
      if (Tin.uten_type() == UTenType.Dense) {
        _Rsvd_truncate_Dense_UT(outCyT, Tin, keepdim, err, is_U, is_vT, return_err, mindim,
                                oversampling_summand, oversampling_factor, power_iteration, seed);
        // } else if (Tin.uten_type() == UTenType.Block) {
        //   _Rsvd_truncate_Block_UT(outCyT, Tin, keepdim, err, is_U, is_vT,
        //   return_err, mindim);
      } else {
        cytnx_error_msg(true, "[ERROR][Rsvd_truncate] only Dense UniTensors are supported.%s",
                        "\n");
      }
      return outCyT;

    }  // Rsvd_truncate

  }  // namespace linalg
}  // namespace cytnx
#endif  // BACKEND_TORCH
