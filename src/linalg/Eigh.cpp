#include "linalg.hpp"

#include <iostream>
#include "Tensor.hpp"
using namespace std;

#ifdef BACKEND_TORCH
#else
  #include "../backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {
    std::vector<Tensor> Eigh(const Tensor &Tin, const bool &is_V, const bool &row_v) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[Eigh] error, Eigh can only operate on rank-2 Tensor.%s", "\n");
      // cytnx_error_msg(!Tin.is_contiguous(), "[Eigh] error tensor must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1],
                      "[Eigh] error, Eigh should accept a square matrix.%s", "\n");

      // std::cout << Tin << std::endl;
      Tensor in = Tin.contiguous();
      if (Tin.dtype() > Type.Float) in = in.astype(Type.Double);

      Tensor S, V;
      S.Init({in.shape()[0]}, in.dtype() <= 2 ? in.dtype() + 2 : in.dtype(),
             in.device());  // if type is complex, S should be real
      if (is_V) {
        V.Init(in.shape(), in.dtype(), in.device());
      }

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Eigh_ii[in.dtype()](in._impl->storage()._impl,
                                                        S._impl->storage()._impl,
                                                        V._impl->storage()._impl, in.shape()[0]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_V) {
          out.push_back(V);
          if (!row_v) {
            if (out.back().dtype() == Type.ComplexFloat ||
                out.back().dtype() == Type.ComplexDouble) {
              out.back().permute_({1, 0}).contiguous_();
              out.back().Conj_();
              // std::cout << "ok";
            } else
              out.back().permute_({1, 0}).contiguous_();
          }
        }

        return out;

      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(in.device()));
        cytnx::linalg_internal::lii.cuEigh_ii[in.dtype()](in._impl->storage()._impl,
                                                          S._impl->storage()._impl,
                                                          V._impl->storage()._impl, in.shape()[0]);

        std::vector<Tensor> out;
        out.push_back(S);
        if (is_V) {
          out.push_back(V);
          if (!row_v) {
            if (out.back().dtype() == Type.ComplexFloat ||
                out.back().dtype() == Type.ComplexDouble) {
              out.back().permute_({1, 0}).contiguous_();
              out.back().Conj_();
              // std::cout << "ok";
            } else
              out.back().permute_({1, 0}).contiguous_();
          }
        }

        return out;
  #else
        cytnx_error_msg(true, "[Eigh] fatal error,%s",
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
    void _Eigh_Dense_UT(std::vector<cytnx::UniTensor> &outCyT, const UniTensor &Tin,
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

      vector<Tensor> outT = cytnx::linalg::Eigh(tmp, is_V, row_v);
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
    }  //_Eigh_Dense_UT

    std::vector<cytnx::UniTensor> Eigh(const UniTensor &Tin, const bool &is_V, const bool &row_v) {
      // using rowrank to split the bond to form a matrix.
      cytnx_error_msg(Tin.rowrank() < 1 || Tin.rank() == 1,
                      "[Eigh][ERROR] Eigh for UniTensor should have rank>1 and rowrank>0%s", "\n");

      cytnx_error_msg(Tin.is_diag(),
                      "[Eigh][ERROR] Eigh for diagonal UniTensor is trivial and currently not "
                      "support. Use other manipulation.%s",
                      "\n");

      std::vector<UniTensor> outCyT;
      if (Tin.uten_type() == UTenType.Dense) {
        _Eigh_Dense_UT(outCyT, Tin, is_V, row_v);

      } else {
        cytnx_error_msg(true,
                        "[ERROR] Eigh, unsupported type of UniTensor only support (Dense). "
                        "something wrong internal%s",
                        "\n");
      }  // is block form ?

      return outCyT;

    };  // Eigh

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
