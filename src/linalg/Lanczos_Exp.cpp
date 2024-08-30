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

    // project v to u
    static UniTensor _Gram_Schimidt_proj(const UniTensor &v, const UniTensor &u) {
      auto nu = _Dot(u, v);
      auto de = _Dot(u, u);
      auto coe = nu / de;
      return coe * u;
    }

    static UniTensor _Gram_Schimidt(const std::vector<UniTensor> &vs) {
      auto u = vs.at(0).clone();
      double low = -1.0, high = 1.0;
      random::uniform_(u, low, high);
      for (auto &v : vs) {
        u -= _Gram_Schimidt_proj(u, v);
      }
      return u;
    }

    static Tensor _resize_mat(const Tensor &src, const cytnx_uint64 r, const cytnx_uint64 c) {
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

    // BiCGSTAB method to solve the linear equation
    // ref: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
    UniTensor _invert_biCGSTAB(LinOp *Hop, const UniTensor &b, const UniTensor &Tin, const int &k,
                               const double &CvgCrit = 1.0e-12,
                               const unsigned int &Maxiter = 10000) {
      // the operation (I + Hop/k) on A
      auto I_plus_A_Op = [&](UniTensor A) {
        return ((Hop->matvec(A)) / k + A).relabels_(b.labels());
      };
      // the residuals of (b - (I + Hop/k)x)
      auto r = (b - I_plus_A_Op(Tin)).relabels_(b.labels());
      // choose r0_hat = r
      auto r0 = r;
      auto x = Tin;
      // choose p = (r0_hat, r)
      auto p = _Dot(r0, r);
      // choose pv = r
      auto pv = r;

      // to reduce the variables used, replace p[i]->p, p[i-1]->p_old, etc.
      auto p_old = p;
      auto pv_old = pv;
      auto x_old = x;
      auto r_old = r;

      // all of logic here is same
      // as:https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
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

        // update
        pv_old = pv;
        p_old = p;
        x_old = x;
        r_old = r;
      }
      return x;
    }

    // ref:  https://doi.org/10.48550/arXiv.1111.1491
    void _Lanczos_Exp_Ut_positive(UniTensor &out, LinOp *Hop, const UniTensor &Tin,
                                  const double &CvgCrit, const unsigned int &Maxiter,
                                  const bool &verbose) {
      double delta = CvgCrit;
      int k = static_cast<int>(std::log(1.0 / delta));
      k = k < Maxiter ? k : Maxiter;
      auto Op_apprx_norm =
        static_cast<double>(Hop->matvec(Tin).get_block_().flatten().Norm().item().real());
      double eps1 = std::exp(-(k * std::log(k) + std::log(1.0 + Op_apprx_norm)));

      std::vector<UniTensor> vs;
      Tensor as = zeros({(cytnx_uint64)k + 1, (cytnx_uint64)k + 1}, Hop->dtype(), Hop->device());

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
          as.at({(cytnx_uint64)j, (cytnx_uint64)i}) = _Dot(Vs.at(j), w);
        }
        // Define wp[i] = w[i] - \sum_{j=0}^{j=i} {a[j,i]v[j]}
        auto wp = w;
        for (int j = 0; j <= i; j++) {
          wp -= as.at({(cytnx_uint64)j, (cytnx_uint64)i}) * Vs.at(j);
        }
        // Let a[i+1, i] = |wp[i]|, v[i+1]=wp[i] / a[i+1, i]
        auto b = std::sqrt(double(_Dot(wp, wp).real()));
        if (i < k) {
          as.at({(cytnx_uint64)i + 1, (cytnx_uint64)i}) = b;
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

    void _Lanczos_Exp_Ut(UniTensor &out, LinOp *Hop, const UniTensor &T, Scalar tau,
                         const double &CvgCrit, const unsigned int &Maxiter, const bool &verbose) {
      const double beta_tol = 1.0e-6;
      std::vector<UniTensor> vs;
      cytnx_uint32 vec_len = Hop->nx();
      cytnx_uint32 imp_maxiter = std::min(Maxiter, vec_len + 1);
      Tensor Hp = zeros({imp_maxiter, imp_maxiter}, Hop->dtype(), Hop->device());

      Tensor B_mat;
      // prepare initial tensor and normalize
      auto v = T.clone();
      auto v_nrm = std::sqrt(double(_Dot(v, v).real()));
      v = v / v_nrm;

      // first iteration
      auto wp = (Hop->matvec(v)).relabels_(v.labels());
      auto alpha = _Dot(wp, v);
      Hp.at({0, 0}) = alpha;
      auto w = (wp - alpha * v).relabels_(v.labels());

      // prepare U
      auto Vk_shape = v.shape();
      Vk_shape.insert(Vk_shape.begin(), 1);
      auto Vk = v.get_block().reshape(Vk_shape);
      std::vector<UniTensor> Vs;
      Vs.push_back(v);
      UniTensor v_old;
      Tensor Hp_sub;

      for (int i = 1; i < imp_maxiter; ++i) {
        if (verbose) {
          std::cout << "Lancos iteration:" << i << std::endl;
        }
        auto beta = std::sqrt(double(_Dot(w, w).real()));
        v_old = v.clone();
        if (beta > beta_tol) {
          v = (w / beta).relabels_(v.labels());
        } else {  // beta too small -> the norm of new vector too small. This vector cannot span the
                  // new dimension
          if (verbose) {
            std::cout << "beta too small, pick another vector." << i << std::endl;
          }
          // pick a new vector perpendicular to all vector in Vs
          v = _Gram_Schimidt(Vs).relabels_(v.labels());
          auto v_norm = _Dot(v, v);
          // if the picked vector also too small, break and construct expH
          if (abs(v_norm) <= beta_tol) {
            if (verbose) {
              std::cout << "All vector form the space. Break." << i << std::endl;
            }
            break;
          }
          v = v / v_norm;
        }
        Vk.append(v.get_block_().contiguous());
        Vs.push_back(v);
        Hp.at({(cytnx_uint64)i, (cytnx_uint64)i - 1}) =
          Hp.at({(cytnx_uint64)i - 1, (cytnx_uint64)i}) = beta;
        wp = (Hop->matvec(v)).relabels_(v.labels());
        alpha = _Dot(wp, v);
        Hp.at({(cytnx_uint64)i, (cytnx_uint64)i}) = alpha;
        w = (wp - alpha * v - beta * v_old).relabels_(v.labels());

        // Converge check
        Hp_sub = _resize_mat(Hp, i + 1, i + 1);
        // We use ExpM since H*tau may not be Hermitian if tau is complex.
        B_mat = linalg::ExpM(Hp_sub * tau);
        // Set the error as the element of bottom left of the exp(H_sub*tau)
        auto error = abs(B_mat.at({(cytnx_uint64)i, 0}));
        if (error < CvgCrit || i == imp_maxiter - 1) {
          if (i == imp_maxiter - 1 && error > CvgCrit) {
            cytnx_warning_msg(
              true,
              "[WARNING][Lanczos_Exp] Fail to converge at eigv [%d], try increasing "
              "maxiter?\n Note:: ignore if this is intended.%s",
              imp_maxiter, "\n");
          }
          break;
        }
      }
      // std::cout << B_mat;

      // Let V_k be the n × (k + 1) matrix whose columns are v[0],...,v[k] respectively.
      UniTensor Vk_ut(Vk);
      Vk_ut.set_rowrank_(1);
      auto VkDag_ut = Vk_ut.Dagger();
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
      auto Vk_labels = v.labels();
      Vk_labels.insert(Vk_labels.begin(), label_kl);
      Vk_ut.relabels_(Vk_labels);
      auto VkDag_labels = v.labels();
      VkDag_labels.push_back(label_kr);
      VkDag_ut.relabels_(VkDag_labels);

      out = Contracts({T, VkDag_ut, B}, "", true);
      out = Contract(out, Vk_ut);
      out.set_rowrank_(v.rowrank());
    }

    // Lanczos_Exp
    UniTensor Lanczos_Exp(LinOp *Hop, const UniTensor &Tin, const Scalar &tau,
                          const double &CvgCrit, const unsigned int &Maxiter, const bool &verbose) {
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

      //_Lanczos_Exp_Ut_positive(out, Hop, v0, _cvgcrit, Maxiter, verbose);
      _Lanczos_Exp_Ut(out, Hop, v0, tau, _cvgcrit, Maxiter, verbose);

      return out;

    }  // Lanczos_Exp entry point

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
