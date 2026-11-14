#include "linalg.hpp"
#include "Generator.hpp"
#include "random.hpp"
#include "Tensor.hpp"
#include "LinOp.hpp"

#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include "UniTensor.hpp"
#include "utils/vec_print.hpp"
#include <iomanip>

#include "Lanczos_Gnd_breakdown.hpp"

namespace cytnx {
  namespace linalg {
    typedef Accessor ac;

    // <A|B>
    static Scalar _Dot(const UniTensor &A, const UniTensor &B) {
      // For a fermionic tensor the bra-ket contraction carries sign flips; applying
      // fermion_twists() to the dagger (bra) operand cancels them so the result is the true scalar
      // product. For bosonic/dense tensors fermion_twists() is not defined, so use the plain
      // contraction.
      if (A.uten_type() == UTenType.BlockFermionic)
        return Contract(A.Dagger().fermion_twists(), B).item();
      return Contract(A.Dagger(), B).item();
    }

    // Number of stored elements, i.e. the dimension of the vector space the UniTensor lives in.
    static cytnx_uint64 stored_elements(const UniTensor &v) {
      if (!v.is_blockform()) return v.get_block_().storage().size();
      cytnx_uint64 count = 0;
      for (const Tensor &block : v.get_blocks_()) count += block.storage().size();
      return count;
    }

    void _Lanczos_Gnd_general_Ut(std::vector<UniTensor> &out, LinOp *Hop, const UniTensor &Tin,
                                 const bool &is_V, const double &CvgCrit,
                                 const unsigned int &Maxiter, const bool &verbose) {
      out.clear();
      std::vector<UniTensor> psi_s;
      //[require] Tin should be provided!

      // Frobenius norm is sign-flip independent, so it is the correct vector norm for fermionic
      // tensors too (and equals sqrt(<Tin|Tin>) for bosonic).
      double Norm = double(Tin.norm());
      cytnx_error_msg(!(Norm > 0), "[ERROR][Lanczos] the initial vector Tin must be nonzero.%s",
                      "\n");

      UniTensor psi_1 = Tin / Norm;
      psi_1.contiguous_();
      // Pin the Krylov vectors to the sign frame with all signflips applied and the vector
      // arithmetic below stay consistent; no-op for bosonic/dense tensors.
      psi_1.apply_();
      if (is_V) {
        psi_s.push_back(psi_1);
      }

      UniTensor psi_0;  // = cytnx::zeros({psi_1.shape()[0]},psi_1.dtype(),Tin.device());
      UniTensor new_psi;
      bool cvg_fin = false;

      // declare variables, A,B should be real if LinOp is hermitian!
      const unsigned int coeff_dtype = Type.to_real(Hop->dtype());
      const double breakdown_tol = lanczos_gnd_breakdown_tolerance(coeff_dtype);
      Tensor As = zeros({1}, coeff_dtype, Tin.device());
      Tensor Bs = As.clone();
      Scalar E;

      // temporary:
      std::vector<Tensor> tmpEsVs;

      // i=0
      //-------------------------------------------------
      new_psi = Hop->matvec(psi_1);
      new_psi.apply_();  // resolve pending fermionic signflip; no-op for bosonic/dense

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

      // At most dim linearly independent Krylov vectors exist. The dimension is taken from the
      // vectors themselves rather than from the user-supplied LinOp::nx().
      const cytnx_uint64 dim = std::max(stored_elements(psi_1), stored_elements(new_psi));

      auto alpha = _Dot(new_psi, psi_1).real();
      As(0) = alpha;
      new_psi -= alpha * psi_1;
      double beta = double(new_psi.norm());
      Bs(0) = beta;
      // Size of the Lanczos coefficients seen so far, the scale a residual norm is compared to.
      double coeff_scale = std::abs(double(alpha));

      if (dim == 1 || beta <= breakdown_tol * coeff_scale) {
        // The initial vector spans an invariant subspace: the Krylov matrix is [As(0)], with
        // eigenvalue As(0) and eigenvector (1).
        cvg_fin = true;
        tmpEsVs = {As.clone(), ones({1, 1}, coeff_dtype, Tin.device())};
      } else {
        psi_0 = psi_1;
        new_psi /= beta;
        psi_1 = new_psi;
        if (is_V) {
          psi_s.push_back(psi_1);
        }
        E = alpha;
        Scalar Ediff;

        ///---------------------------

        const unsigned int max_krylov_dim =
          static_cast<unsigned int>(std::min<cytnx_uint64>(Maxiter, dim));

        // iteration LZ:
        for (unsigned int i = 1; i < max_krylov_dim; i++) {
          new_psi = Hop->matvec(psi_1);
          new_psi.apply_();  // resolve pending fermionic signflip; no-op for bosonic/dense
          alpha = _Dot(new_psi, psi_1).real();
          As.append(alpha);
          new_psi -= (alpha * psi_1 + beta * psi_0);

          // diagonalize:
          tmpEsVs = linalg::Tridiag(As, Bs, true, true);

          coeff_scale = std::max(coeff_scale, std::abs(double(alpha)) + beta);
          beta = double(new_psi.norm());
          Bs.append(beta);

          // Stop when the residual is at roundoff level (the Krylov space is invariant) or when
          // the Krylov space has become the whole vector space.
          if (beta <= breakdown_tol * coeff_scale || i + 1 == dim) {
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
      }

      if (cvg_fin == false && verbose) {
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
          cytnx_warning_msg(true,
                            "[WARNING][CvgCrit] for float precision type, CvgCrit cannot be "
                            "smaller than 1e-7, and is automatically raised to 1e-7.%s",
                            "\n");
        }
      }

      _Lanczos_Gnd_general_Ut(out, Hop, v0, is_V, _cvgcrit, Maxiter, verbose);

      return out;

    }  // Lanczos_Gnd entry point

  }  // namespace linalg
}  // namespace cytnx
