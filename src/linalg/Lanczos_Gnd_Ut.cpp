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

    void _Lanczos_Gnd_general_Ut(std::vector<UniTensor> &out, LinOp *Hop, const UniTensor &Tin,
                                 const bool &is_V, const double &CvgCrit,
                                 const unsigned int &Maxiter, const bool &verbose) {
      out.clear();
      std::vector<UniTensor> psi_s;
      //[require] Tin should be provided!

      double Norm = double(Contract(Tin, Tin.Dagger()).item().real());
      Norm = std::sqrt(Norm);

      UniTensor psi_1 = Tin / Norm;
      psi_1.contiguous_();
      if (is_V) {
        psi_s.push_back(psi_1);
      }

      UniTensor psi_0;  // = cytnx::zeros({psi_1.shape()[0]},psi_1.dtype(),Tin.device());
      UniTensor new_psi;
      bool cvg_fin = false;

      // declare variables, A,B should be real if LinOp is hermitian!
      Tensor As = zeros({1}, Hop->dtype() < 3 ? Hop->dtype() + 2 : Hop->dtype(), Tin.device());
      Tensor Bs = As.clone();
      Scalar E;

      // temporary:
      std::vector<Tensor> tmpEsVs;

      // i=0
      //-------------------------------------------------
      new_psi = Hop->matvec(psi_1);

      /*
         checking if the output match input:
      */
      cytnx_error_msg(new_psi.labels().size() != psi_1.labels().size(),
                      "[ERROR] LinOp.matvec(UniTensor) -> UniTensor the output should have same "
                      "labels and shape as input!%s",
                      "\n");
      cytnx_error_msg(new_psi.labels() != psi_1.labels(),
                      "[ERROR] LinOp.matvec(UniTensor) -> UniTensor the output should have same "
                      "labels and shape as input!%s",
                      "\n");

      auto alpha = _Dot(new_psi, psi_1).real();
      As(0) = alpha;
      new_psi -= alpha * psi_1;
      auto beta = new_psi.Norm().item();
      Bs(0) = beta;
      psi_0 = psi_1;
      new_psi /= beta;
      psi_1 = new_psi;
      if (is_V) {
        psi_s.push_back(psi_1);
      }
      E = alpha;
      Scalar Ediff;

      ///---------------------------

      // iteration LZ:
      for (unsigned int i = 1; i < Maxiter; i++) {
        new_psi = Hop->matvec(psi_1);
        alpha = _Dot(new_psi, psi_1).real();
        As.append(alpha);
        new_psi -= (alpha * psi_1 + beta * psi_0);

        // diagonalize
        try {
          // diagonalize:
          auto tmptmp = linalg::Tridiag(As, Bs, true, true, true);
          tmpEsVs = tmptmp;
        } catch (std::logic_error le) {
          std::cout << "[WARNING] Lanczos_Gnd -> Tridiag error: \n";
          std::cout << le.what() << std::endl;
          std::cout << "Lanczos continues automatically." << std::endl;
          break;
        }

        beta = new_psi.Norm().item();
        Bs.append(beta);

        if (beta == 0) {
          cvg_fin = true;
          break;
        }

        psi_0 = psi_1;
        psi_1 = new_psi / beta;
        if (is_V) {
          psi_s.push_back(psi_1);
        }
        Ediff = abs(E - tmpEsVs[0].storage().at(0));
        if (verbose) {
          printf("iter[%d] Enr: %11.14f, diff from last iter: %11.14f\n", i, double(E),
                 double(Ediff));
        }

        // chkf = true;
        if (Ediff < CvgCrit) {
          cvg_fin = true;
          break;
        }
        E = tmpEsVs[0].storage().at(0);

      }  // iteration

      if (cvg_fin == false) {
        cytnx_warning_msg(true,
                          "[WARNING] iteration not converge after Maxiter!.\n :: Note :: ignore if "
                          " this is intended %s",
                          "\n");
      }
      out.push_back(UniTensor(tmpEsVs[0](0), false, 0));

      if (is_V) {
        UniTensor eV;
        Storage kryVg = tmpEsVs[1](0).storage();
        tmpEsVs.pop_back();

        eV = kryVg.at(0) * psi_s.at(0);
        for (unsigned int n = 1; n < tmpEsVs[0].shape()[0]; n++) {
          eV += kryVg.at(n) * psi_s.at(n);
        }

        out.push_back(eV);
      }
    }

    // Lanczos
    std::vector<UniTensor> Lanczos_Gnd_Ut(LinOp *Hop, const UniTensor &Tin, const double &CvgCrit,
                                          const bool &is_V, const bool &verbose,
                                          const unsigned int &Maxiter) {
      // check type:
      cytnx_error_msg(
        !Type.is_float(Hop->dtype()),
        "[ERROR][Lanczos] Lanczos can only accept operator with floating types (complex/real)%s",
        "\n");

      // check criteria and maxiter:
      cytnx_error_msg(CvgCrit <= 0, "[ERROR][Lanczos] converge criteria must >0%s", "\n");
      cytnx_error_msg(Maxiter < 2, "[ERROR][Lanczos] Maxiter must >1%s", "\n");

      UniTensor v0;
      v0 = Tin.astype(Hop->dtype());

      std::vector<UniTensor> out;

      double _cvgcrit = CvgCrit;

      if (Hop->dtype() == Type.Float || Hop->dtype() == Type.ComplexFloat) {
        if (_cvgcrit < 1.0e-7) {
          _cvgcrit = 1.0e-7;
          cytnx_warning_msg(_cvgcrit < 1.0e-7,
                            "[WARNING][CvgCrit] for float precision type, CvgCrit cannot exceed "
                            "it's own type precision limit 1e-7, and it's auto capped to 1.0e-7.%s",
                            "\n");
        }
      }

      _Lanczos_Gnd_general_Ut(out, Hop, v0, is_V, _cvgcrit, Maxiter, verbose);

      return out;

    }  // Lanczos_Gnd entry point

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
