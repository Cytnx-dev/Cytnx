#include "linalg.hpp"
#include "Generator.hpp"
#include "random.hpp"
#include "Tensor.hpp"
#include "LinOp.hpp"

#include <cfloat>
#include <algorithm>
namespace cytnx {
  namespace linalg {
    typedef Accessor ac;

    // resize the matrix (2-rank tensor)
    static Tensor ResizeMat(const Tensor &src, const cytnx_uint64 r, const cytnx_uint64 c) {
      const auto min_r = std::min(r, src.shape()[0]);
      const auto min_c = std::min(c, src.shape()[1]);
      // Tensor dst = src[{ac::range(0,min_r),ac::range(0,min_c)}];

      Tensor dst = Tensor({min_r, min_c}, src.dtype(), src.device(), false);
      char *tgt = (char *)dst.storage().data();
      char *csc = (char *)src.storage().data();
      unsigned long long Offset_csc = Type.typeSize(src.dtype()) * src.shape()[1];
      unsigned long long Offset_tgt = Type.typeSize(src.dtype()) * min_c;
      for (auto i = 0; i < min_r; ++i) {
        memcpy(tgt + Offset_tgt * i, csc + Offset_csc * i, Type.typeSize(src.dtype()) * min_c);
      }

      return dst;
    }

    // Get the indices of the first few order element
    std::vector<cytnx_int32> GetFstFewOrderElemIdices(const Tensor &tens, const std::string &which,
                                                      const cytnx_int64 k) {
      char large_or_small = which[0];  //'S' or 'L'
      char metric_type = which[1];  //'M', 'R' or 'I'

      // get the metric distance
      auto len = tens.shape()[0];
      std::vector<Scalar> vec;
      vec = std::vector<Scalar>(len, 0);
      if (metric_type == 'M') {
        for (int i = 0; i < len; ++i) vec[i] = abs(tens.storage().at(i));
      } else if (metric_type == 'R') {
        for (int i = 0; i < len; ++i) vec[i] = tens.storage().at(i).real();
      } else if (metric_type == 'I') {
        for (int i = 0; i < len; ++i) vec[i] = tens.storage().at(i).imag();
      } else {
        ;
      }  // never

      // smallest or largest
      bool is_small = (large_or_small == 'S');
      Scalar init_scalar = is_small ? Scalar::maxval(Type.Double) : 0;
      auto indices = std::vector<cytnx_int32>(k, -1);
      for (cytnx_int32 i = 0; i < k; ++i) {
        auto itr = is_small ? std::min_element(vec.begin(), vec.end())
                            : std::max_element(vec.begin(), vec.end());
        indices[i] = static_cast<cytnx_int32>(itr - vec.begin());
        *itr = init_scalar;
      }
      return indices;
    }

    bool IsEigvalCvg(const std::vector<Scalar> &eigvals, const std::vector<Scalar> &eigvals_old,
                     const double cvg_crit) {
      for (cytnx_int32 i = 0; i < eigvals.size(); ++i) {
        auto err = abs(eigvals[i] - eigvals_old[i]);
        if (err >= cvg_crit) return false;
      }
      return true;
    }

    // check the residule |Mv - ev| is converged.
    bool IsResiduleSmallEnough(LinOp *Hop, const std::vector<Tensor> &eigvecs,
                               const std::vector<Scalar> &eigvals, const double cvg_crit) {
      for (cytnx_int32 i = 0; i < eigvals.size(); ++i) {
        auto eigvec = eigvecs[i];
        auto eigval = eigvals[i];
        auto resi = (Hop->matvec(eigvec) - eigval * eigvec).Norm().item();
        if (resi >= cvg_crit) return false;
      }
      return true;
    }

    std::vector<Tensor> GetEigTens(const std::vector<Tensor> qs, Tensor eigvec_in_kryv,
                                   std::vector<cytnx_int32> max_indices) {
      auto k = max_indices.size();
      cytnx_int64 krydim = eigvec_in_kryv.shape()[0];
      auto P_inv = InvM(eigvec_in_kryv).Conj();
      auto eigTens_s = std::vector<Tensor>(k, Tensor());
      for (cytnx_int32 ik = 0; ik < k; ++ik) {
        auto maxIdx = max_indices[ik];
        auto eigTens = zeros(qs[0].shape(), Type.ComplexDouble);
        for (cytnx_int64 i = 0; i < krydim; ++i) {
          eigTens += P_inv[{i, maxIdx}] * qs[i];
        }
        eigTens /= eigTens.Norm().item();
        eigTens_s[ik] = eigTens;
      }
      return eigTens_s;
    }

    void _Arnoldi(std::vector<Tensor> &out, LinOp *Hop, const Tensor &T_init,
                  const std::string which, const cytnx_uint64 &maxiter, const double &CvgCrit,
                  const cytnx_uint64 &k, const bool &is_V, const bool &verbose) {
      auto vec_len = T_init.shape()[0];
      const cytnx_uint64 imp_maxiter = std::min(maxiter, vec_len + 1);
      const cytnx_complex128 unit_complex = 1.0;
      // out[0]:eigenvalues, out[1]:eigentensors
      out[0] = zeros({k}, Type.ComplexDouble);  // initialize
      auto eigvals = std::vector<Scalar>(k, Scalar());
      std::vector<Tensor> eigTens_s;
      Tensor kry_mat_buffer =
        cytnx::zeros({imp_maxiter + 1, imp_maxiter + 1}, Hop->dtype(), Hop->device());
      bool is_cvg = false;
      auto eigvals_old = std::vector<Scalar>(k, Scalar::maxval(Type.Double));
      std::vector<Tensor> buffer;
      buffer.push_back(T_init);
      buffer[0] = buffer[0] / buffer[0].Norm().item();  // normalized q1

      // start arnoldi iteration
      for (auto i = 1; i < imp_maxiter; i++) {
        cytnx_uint64 krydim = i;
        auto nextTens = Hop->matvec(buffer[i - 1]).astype(Hop->dtype());
        buffer.push_back(nextTens);
        for (cytnx_uint32 j = 0; j < krydim; j++) {
          auto h = Vectordot(buffer[i], buffer[j], true).Conj();
          kry_mat_buffer[{i - 1, j}] = h;
          buffer[i] -= h * buffer[j];
        }
        auto h = buffer[i].Norm().item();
        kry_mat_buffer[{i - 1, i}] = h;
        buffer[i] /= h;
        Tensor kry_mat = ResizeMat(kry_mat_buffer, krydim, krydim);

        // call Eig to get eigenvalues
        auto eigs = Eig(kry_mat, true, true);
        // get first few order of eigenvlues
        std::vector<cytnx_int32> maxIndices = GetFstFewOrderElemIdices(eigs[0], which, k);
        for (cytnx_int32 ik = 0; ik < k; ++ik) {
          auto maxIdx = maxIndices[ik];
          eigvals[ik] = eigs[0].storage().at(maxIdx);
        }

        // check converged
        bool is_eigval_cvg = IsEigvalCvg(eigvals, eigvals_old, CvgCrit);
        if (is_eigval_cvg || i == imp_maxiter - 1) {
          eigTens_s = GetEigTens(buffer, eigs[1], maxIndices);
          bool is_res_small_enough = IsResiduleSmallEnough(Hop, eigTens_s, eigvals, CvgCrit);
          if (is_res_small_enough) {
            is_cvg = true;
            break;
          }
        }
        eigvals_old = eigvals;
      }  // Arnoldi iteration
      buffer.clear();

      // set output
      for (cytnx_int32 ik = 0; ik < k; ++ik) out[0][{ac(ik)}] = eigvals[ik];
      if (is_V)  // if need output eigentensors
      {
        out[1] = eigTens_s[0];
        if (eigTens_s.size() > 1)  // if k > 1, append the eigenvector as a single Tensor'.
        {
          out[1].reshape_({1, -1});
          for (cytnx_uint64 i = 1; i < eigTens_s.size(); ++i) out[1].append(eigTens_s[i]);
        }
      }
    }

    std::vector<Tensor> Arnoldi(LinOp *Hop, const Tensor &T_init, const std::string which,
                                const cytnx_uint64 &maxiter, const double &cvg_crit,
                                const cytnx_uint64 &k, const bool &is_V, const bool &verbose) {
      // check type:
      cytnx_error_msg(
        !Type.is_float(Hop->dtype()),
        "[ERROR][Arnoldi] Arnoldi can only accept operator with floating types (complex/real)%s",
        "\n");

      // check which
      std::vector<std::string> accept_which = {"LM", "LR", "LI", "SM", "SR", "SI"};
      if (std::find(accept_which.begin(), accept_which.end(), which) == accept_which.end()) {
        cytnx_error_msg(true,
                        "[ERROR][Arnoldi] 'which' should be 'LM', 'LR, 'LI'"
                        ", 'SM', 'SR, 'SI'",
                        "\n");
      }

      /// check k
      cytnx_error_msg(k < 1, "[ERROR][Arnoldi] k should be >0%s", "\n");
      cytnx_error_msg(k > Hop->nx(),
                      "[ERROR][Arnoldi] k can only be up to total dimension of input vector D%s",
                      "\n");

      // check Tin should be rank-1:
      auto _T_init = T_init.clone();
      if (T_init.dtype() == Type.Void) {
        _T_init =
          cytnx::random::normal({Hop->nx()}, Hop->dtype(), Hop->device());  // randomly initialize.
      } else {
        cytnx_error_msg(T_init.shape().size() != 1, "[ERROR][Arnoldi] Tin should be rank-1%s",
                        "\n");
        cytnx_error_msg(T_init.shape()[0] != Hop->nx(),
                        "[ERROR][Arnoldi] Tin should have dimension consistent with Hop: [%d] %s",
                        Hop->nx(), "\n");
        _T_init = T_init.astype(Hop->dtype());
      }

      cytnx_error_msg(cvg_crit <= 0, "[ERROR][Arnoldi] cvg_crit should be > 0%s", "\n");
      double _cvgcrit = cvg_crit;
      cytnx_uint64 output_size = is_V ? 2 : 1;
      auto out = std::vector<Tensor>(output_size, Tensor());
      _Arnoldi(out, Hop, T_init, which, maxiter, cvg_crit, k, is_V, verbose);
      return out;
    }

  }  // namespace linalg
}  // namespace cytnx
