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

    // <A|B>
    static Scalar _Dot(const UniTensor &A, const UniTensor &B) {
      return Contract(A.Dagger(), B).item();
    }

    // BiCGSTAB method to solve the linear equation
    // ref: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
    UniTensor _invert_biCGSTAB(LinOp *Hop, const UniTensor &b, const UniTensor &Tin, const int &k,
                               const double &CvgCrit = 1.0e-12,
                               const unsigned int &Maxiter = 10000) {
	  //the operation (I + Hop/k) on A
      auto I_plus_A_Op = [&](UniTensor A) {
        return ((Hop->matvec(A)) / k + A).relabels_(b.labels());
      };
	  //the residuals of (b - (I + Hop/k)x)
      auto r = (b - I_plus_A_Op(Tin)).relabels_(b.labels());
	  //choose r0_hat = r
      auto r0 = r;
      auto x = Tin;
	  //choose p = (r0_hat, r)
      auto p = _Dot(r0, r);
	  //choose pv = r
      auto pv = r;

	  //to reduce the variables used, replace p[i]->p, p[i-1]->p_old, etc.
      auto p_old = p;
      auto pv_old = pv;
      auto x_old = x;
      auto r_old = r;

	  //all of logic here is same as:https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
      for (int i = 1; i < Maxiter; ++i) {
        auto v = I_plus_A_Op(pv_old);
        auto a = p_old / _Dot(r0, v);
        auto h = (x_old + a * pv_old).relabels_(b.labels());
        auto s = (r_old - a * v).relabels_(b.labels());
        if (abs(_Dot(s, s)) < CvgCrit) {
          x = h;
          break;
        }
        auto t = I_plus_A_Op(s);
        auto w = _Dot(t, s) / _Dot(t, t);
        x = (h + w * s).relabels_(b.labels());
        r = (s - w * t).relabels_(b.labels());
        if (abs(_Dot(r, r)) < CvgCrit) {
          break;
        }
        auto p = _Dot(r0, r);
        auto beta = (p / p_old) * (a / w);
        pv = (r + beta * (pv_old - w * v)).relabels_(b.labels());

		//update
        pv_old = pv;
        p_old = p;
        x_old = x;
        r_old = r;
      }
      return x;
    }

    // ref:  https://doi.org/10.48550/arXiv.1111.1491
    void _Lanczos_Exp_Ut(UniTensor &out, LinOp *Hop, const UniTensor &Tin, const double &CvgCrit,
                         const unsigned int &Maxiter, const bool &verbose) {
      double delta = CvgCrit;
      int k = static_cast<int>(std::log(1.0 / delta));
      k = k < Maxiter ? k : Maxiter;
      auto Op_apprx_norm =
        static_cast<double>(Hop->matvec(Tin).get_block_().flatten().Norm().item().real());
      double eps1 = std::exp(-(k * std::log(k) + std::log(1.0 + Op_apprx_norm)));

      std::vector<UniTensor> vs;
      Tensor as = zeros({k + 1, k + 1}, Hop->dtype(), Hop->device());

      // Initialized v0 = v
      auto v = Tin;
      auto v0 = v;
      auto Vk_shape = v0.shape();
      Vk_shape.insert(Vk_shape.begin(), 1);
      auto Vk = v0.get_block().reshape(Vk_shape);
      std::vector<UniTensor> Vs;
      Vs.push_back(v);

      // For i = 0 to k
      for (int i = 0; i <= k; ++i) {
        // Call the procedure Invert_A (v[i], k, eps1). The procedure returns a vector w[i], such
        // that,
        // |(I + A / k )^(−1) v[i] − w[i]| ≤ eps1 |v[i]| .
        auto w = _invert_biCGSTAB(Hop, v, v, k, eps1);
        // auto resi = ((Hop->matvec(w))/k + w).relabels_(v.labels()) - v;

        // For j = 0 to i
        for (int j = 0; j <= i; ++j) {
          // Let a[j,i] = <v[j], w[i]>
          as.at({j, i}) = _Dot(Vs.at(j), w);
        }
        // Define wp[i] = w[i] - \sum_{j=0}^{j=i} {a[j,i]v[j]}
        auto wp = w;
        for (int j = 0; j <= i; j++) {
          wp -= as.at({j, i}) * Vs.at(j);
        }
        // Let a[i+1, i] = |wp[i]|, v[i+1]=wp[i] / a[i+1, i]
        auto b = std::sqrt(double(_Dot(wp, wp).real()));
        if (i < k) {
          as.at({i + 1, i}) = b;
          v = wp / b;
          Vk.append(v.get_block_());
          Vs.push_back(v);
        }
        // For j = i+2 to k
        //   Let a[j,i] = 0
      }

      // Let V_k be the n × (k + 1) matrix whose columns are v[0],...,v[k] respectively.
      UniTensor Vk_ut(Vk);
      Vk_ut.set_rowrank_(1);
      auto VkDag_ut = Vk_ut.Dagger();
      // Let T_k be the (k + 1) × (k + 1) matrix a[i,j] i,j is {0,...,k} and Tk_hat = 1 / 2
      // (Tk^Dagger  + Tk).
      auto asT = as.permute({1, 0}).Conj().contiguous();
      auto Tk_hat = 0.5 * (asT + as);
      // Compute B = exp k · (I − Tk_hat^(−1) ) exactly and output the vector V_k*B*V_k^Dagger v.
      auto I = eye(k + 1);
      auto B_mat = linalg::ExpH(k * (I - linalg::InvM(Tk_hat)));
      /*
       *    |||
       *  |-----|
       *  | out |        =
       *  |_____|
       *
       *
       *    |||
       *  |-----|
       *  | V_k |
       *  |_____|
       *     |    kl:(k+1) * (k + 1)
       *     |
       *  |-----|
       *  |  B  |
       *  |_____|
       *     |    kr:(k+1) * (k + 1)
       *     |
       *  |------------|
       *  | V_k^Dagger |
       *  |____________|
       *    |||
       *  |-----|
       *  |  v0 |
       *  |_____|
       *
       */
      auto B = UniTensor(B_mat, false, 1, {"cytnx_internal_label_kl", "cytnx_internal_label_kr"});
      auto label_kl = B.labels()[0];
      auto label_kr = B.labels()[1];
      auto Vk_labels = v0.labels();
      Vk_labels.insert(Vk_labels.begin(), label_kl);
      Vk_ut.relabels_(Vk_labels);
      auto VkDag_labels = v0.labels();
      VkDag_labels.push_back(label_kr);
      VkDag_ut.relabels_(VkDag_labels);

      // Vk_ut.print_diagram();
      // VkDag_ut.print_diagram();
      // v0.print_diagram();
      // B.print_diagram();

      out = Contracts({v0, VkDag_ut, B}, "", true);
      out = Contract(out, Vk_ut);
      out.set_rowrank_(v0.rowrank());
    }

    // Lanczos_Exp
    UniTensor Lanczos_Exp(LinOp *Hop, const UniTensor &Tin, const double &CvgCrit,
                          const unsigned int &Maxiter, const bool &verbose) {
      // check device:
      cytnx_error_msg(Hop->device() != Device.cpu,
                      "[ERROR][Lanczos_Exp] Lanczos_Exp still not sopprot cuda devices.%s", "\n");
      // check type:
      cytnx_error_msg(!Type.is_float(Hop->dtype()),
                      "[ERROR][Lanczos_Exp] Lanczos_Exp can only accept operator with "
                      "floating types (complex/real)%s",
                      "\n");

      cytnx_error_msg(Tin.uten_type() != UTenType.Dense,
                      "[ERROR][Lanczos_Exp] The Block UniTensor type is still not supported.%s",
                      "\n");

      // check criteria and maxiter:
      cytnx_error_msg(CvgCrit <= 0, "[ERROR][Lanczos_Exp] converge criteria must >0%s", "\n");
      cytnx_error_msg(Maxiter < 2, "[ERROR][Lanczos_Exp] Maxiter must >1%s", "\n");

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
            "[WARNING][Lanczos_Exp] for float precision type, CvgCrit cannot exceed "
            "it's own type precision limit 1e-7, and it's auto capped to 1.0e-7.%s",
            "\n");
        }
      }

      _Lanczos_Exp_Ut(out, Hop, v0, _cvgcrit, Maxiter, verbose);

      return out;

    }  // Lanczos_Exp entry point

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
