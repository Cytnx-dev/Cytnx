#include "linalg.hpp"
#include "Generator.hpp"
#include "random.hpp"
#include "Tensor.hpp"
#include "LinOp.hpp"

#include <cfloat>
#include <vector>
#include <cmath>
#include "UniTensor.hpp"
#include "utils/vec_print.hpp"
#include <iomanip>

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  namespace linalg {
    typedef Accessor ac;
    using namespace std;

    namespace {
      // <A|B>
      Scalar Dot_internal(const UniTensor &A, const UniTensor &B) {
        return Contract(A.Dagger(), B).item();
      }

      // project v to u
      UniTensor Gram_Schmidt_proj_internal(const UniTensor &v, const UniTensor &u) {
        auto nu = Dot_internal(u, v);
        auto de = Dot_internal(u, u);
        auto coe = nu / de;
        return coe * u;
      }

      UniTensor Gram_Schmidt_internal(const std::vector<UniTensor> &vs) {
        auto u = vs.at(0).clone();
        double low = -1.0, high = 1.0;
        random::uniform_(u, low, high);
        for (auto &v : vs) {
          u -= Gram_Schmidt_proj_internal(u, v);
        }
        return u;
      }

      Tensor resize_mat_internal(const Tensor &src, const cytnx_uint64 r, const cytnx_uint64 c) {
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

      void Arnoldi_Exp_Ut_internal(UniTensor &out, LinOp *Hop, const UniTensor &T, Scalar tau,
                                   const double &CvgCrit, const unsigned int &Maxiter,
                                   const bool &verbose) {
        const double beta_tol = 1.0e-6;
        cytnx_uint32 vec_len = Hop->nx();
        cytnx_uint32 imp_maxiter = std::min(Maxiter, vec_len + 1);
        Tensor Hp = zeros({imp_maxiter, imp_maxiter}, Hop->dtype(), Hop->device());

        Tensor B_mat;
        // prepare initial tensor and normalize
        auto q = T.clone();
        auto q_nrm = std::sqrt(double(Dot_internal(q, q).real()));
		if (q_nrm < beta_tol) {
		  out = q;
		  random::uniform_(q, -1.0, 1.0, 0);
          q_nrm = std::sqrt(double(Dot_internal(q, q).real()));
		}
        q /= q_nrm;
        // prepare U
        auto Qk_shape = q.shape();
        Qk_shape.insert(Qk_shape.begin(), 1);
        auto Qk = q.get_block().reshape(Qk_shape);
        std::vector<UniTensor> Qs;
        Qs.push_back(q);
        UniTensor q_old;
        Tensor Hp_sub;

        for (int k = 1; k <= imp_maxiter; ++k) {
          if (verbose) {
            std::cout << "Arnoldi iteration:" << k << std::endl;
          }
		  q_old = q;
		  q = (Hop->matvec(q_old)).relabels_(q_old.labels());
		  for (int j = 0; j <= k-1; ++j) {
			Hp.at({j, k-1}) = Dot_internal(Qs[j], q);
			q = (q - Hp.at({j, k-1})*Qs[j]).relabels_(q.labels());
		  }
          q_nrm = std::sqrt(double(Dot_internal(q, q).real()));
		  if (k < imp_maxiter) {
		    Hp.at({k, k-1}) = q_nrm;
		  }
		  if (q_nrm > beta_tol) {
		    q = (q/q_nrm).relabels_(q.labels());
		  } else {
            if (verbose) {
              std::cout << "beta too small, pick another vector." << k << std::endl;
            }
            // pick a new vector perpendicular to all vector in Vs
            q = Gram_Schmidt_internal(Qs).relabels_(q.labels());
            q_nrm = std::sqrt(double(Dot_internal(q, q).real()));
            // if the picked vector also too small, break and construct expH
            if (abs(q_nrm) <= beta_tol) {
              if (verbose) {
                std::cout << "All vector form the space. Break." << k << std::endl;
              }
			  Qs.pop_back();
			  if (k == 1) {
                Hp_sub = resize_mat_internal(Hp, k, k);
		        B_mat = Hp_sub.clone()*tau;
		        Exp_(B_mat);
			  }
              break;
            }
		    q = (q/q_nrm).relabels_(q.labels());
		  }

          // Converge check
          Hp_sub = resize_mat_internal(Hp, k, k);
          // We use ExpM since H*tau may not be Hermitian if tau is complex.
          B_mat = linalg::ExpM(Hp_sub * tau);
          // Set the error as the element of bottom left of the exp(H_sub*tau)
          auto error = abs(B_mat.at({(cytnx_uint64)(k-1), 0}));
          if (error < CvgCrit || k == imp_maxiter) {
            if (k == imp_maxiter && error > CvgCrit && k <= vec_len) {
              cytnx_warning_msg(
                true,
                "[WARNING][Arnoldi_Exp] Fail to converge at eigv [%d], try increasing "
                "maxiter?\n Note:: ignore if this is intended.%s",
                imp_maxiter, "\n");
            }
            break;
          } else {
            Qs.push_back(q);
		  }
		}
		for (int q_i = 1; q_i < Qs.size(); ++q_i) {
          Qk.append(Qs[q_i].get_block_().contiguous_());
		}
        // std::cout << B_mat;

        // Let Q_k be the n × (k) matrix whose columns are v[0],...,v[k-1] respectively.
        UniTensor Qk_ut(Qk);
        Qk_ut.set_rowrank_(1);
        auto QkDag_ut = Qk_ut.Dagger();  // left and right indices are exchanged here!
        /*
         *    |||
         *  |-----|
         *  | out |        =
         *  |_____|
         *
         *
         *    |||
         *  |-----|
         *  | Q_k |
         *  |_____|
         *     |    kl:k * k
         *     |
         *  |-----|
         *  |  B  |
         *  |_____|
         *     |    kr:k * k
         *     |
         *  |------------|
         *  | Q_k^Dagger |
         *  |____________|
         *    |||
         *  |-----|
         *  |  q0 |
         *  |_____|
         *
         */

		//std::cout << B_mat.shape() << std::endl;
        auto B = UniTensor(B_mat, false, 1, {"cytnx_internal_label_kl", "cytnx_internal_label_kr"});
        auto label_kl = B.labels()[0];
        auto label_kr = B.labels()[1];
        auto Qk_labels = q.labels();
        Qk_labels.insert(Qk_labels.begin(), label_kl);
        Qk_ut.relabels_(Qk_labels);
        auto QkDag_labels = q.labels();
        QkDag_labels.push_back(label_kr);
        QkDag_ut.relabels_(QkDag_labels);

        out = Contracts({T, QkDag_ut, B}, "", true);
        out = Contract(out, Qk_ut);
        out.set_rowrank_(q.rowrank());
      }
    }  // unnamed namespace

    // Arnoldi_Exp
    UniTensor Arnoldi_Exp(LinOp *Hop, const UniTensor &Tin, const Scalar &tau,
                          const double &CvgCrit, const unsigned int &Maxiter, const bool &verbose) {
      // check device:
      cytnx_error_msg(Hop->device() != Device.cpu,
                      "[ERROR][Arnoldi_Exp] Arnoldi_Exp still not sopprot cuda devices.%s", "\n");
      // check type:
      cytnx_error_msg(!Type.is_float(Hop->dtype()),
                      "[ERROR][Arnoldi_Exp] Arnoldi_Exp can only accept operator with "
                      "floating types (complex/real)%s",
                      "\n");

      cytnx_error_msg(Tin.uten_type() != UTenType.Dense,
                      "[ERROR][Arnoldi_Exp] The Block UniTensor type is still not supported.%s",
                      "\n");

      // check criteria and maxiter:
      cytnx_error_msg(CvgCrit <= 0, "[ERROR][Arnoldi_Exp] converge criteria must >0%s", "\n");
      cytnx_error_msg(Maxiter < 2, "[ERROR][Arnoldi_Exp] Maxiter must >1%s", "\n");

      // check Tin should be rank-1:

      UniTensor v0;
      v0 = Tin.astype(Hop->dtype());

      UniTensor out;

      double _cvgcrit = CvgCrit;

      if (Hop->dtype() == Type.Float || Hop->dtype() == Type.ComplexFloat) {
        if (_cvgcrit < 1.0e-7) {
          _cvgcrit = 1.0e-7;
          cytnx_warning_msg(
            _cvgcrit < 1.0e-7,
            "[WARNING][Arnoldi_Exp] for float precision type, CvgCrit cannot exceed "
            "it's own type precision limit 1e-7, and it's auto capped to 1.0e-7.%s",
            "\n");
        }
      }

      // Arnoldi_Exp_Ut_internal_positive(out, Hop, v0, _cvgcrit, Maxiter, verbose);
      Arnoldi_Exp_Ut_internal(out, Hop, v0, tau, _cvgcrit, Maxiter, verbose);

      return out;

    }  // Arnoldi_Exp entry point

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
