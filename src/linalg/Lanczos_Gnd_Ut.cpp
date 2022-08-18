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

namespace cytnx {
  namespace linalg {
    typedef Accessor ac;
    using namespace std;

    void unsafe_Sub_(UniTensor &UL, const Scalar &a, const UniTensor &UR) {
      // perform UL -= a*UR) for each blocks.
      //[Warning] 1. This function does not check the Bond mismatch of UL and UR. Use with caution.
      //           2. This function does not check if UL and UR are of the same UTenType!
      for (cytnx_int64 blk = 0; blk < UL.Nblocks(); blk++) {
        UL.get_block_(blk) -= a * UR.get_block_(blk);
      }
    }

    void unsafe_Add_(UniTensor &UL, const Scalar &a, const UniTensor &UR) {
      // perform UL += a*UR) for each blocks.
      //[Warning] 1. This function does not check the Bond mismatch of UL and UR. Use with caution.
      //           2. This function does not check if UL and UR are of the same UTenType!
      for (cytnx_int64 blk = 0; blk < UL.Nblocks(); blk++) {
        UL.get_block_(blk) += a * UR.get_block_(blk);
      }
    }

    void _Lanczos_Gnd_general_Ut(std::vector<UniTensor> &out, LinOp *Hop, const UniTensor &Tin,
                                 const bool &is_V, const double &CvgCrit,
                                 const unsigned int &Maxiter, const bool &verbose) {
      out.clear();
      //[require] Tin should be provided!

      double Norm = double(Contract(Tin, Tin.Dagger()).item().real());
      Norm = std::sqrt(Norm);
      UniTensor psi_1 = Tin / Norm;
      psi_1.contiguous_();
      // UniTensor psi_1 = Tin.clone();
      // psi_1.get_block_()/=Tin.get_block_().Norm();

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

      // psi_1.print_diagram();
      // new_psi.print_diagram();

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

      As(0) = Contract(new_psi.Dagger(), psi_1).item().real();

      // As(0) =
      // linalg::Vectordot(new_psi.get_block_().flatten(),psi_1.get_block_().flatten(),true);

      // cout << (new_psi);
      // new_psi -= As(0)*psi_1;
      unsafe_Sub_(new_psi, As(0).item(), psi_1);

      Bs(0) = new_psi.Norm().item();  // sqrt(Contract(new_psi,new_psi.Dagger()).item().real());
      // Bs(0) = new_psi.get_block_().Norm();

      psi_0 = psi_1;

      new_psi /= Bs(0).item();

      // chekc:

      psi_1 = new_psi;
      E = As(0).item();
      Scalar Ediff;

      ///---------------------------

      // iteration LZ:
      for (unsigned int i = 1; i < Maxiter; i++) {
        // cout << "iter: " << i << "chck:" << endl;
        // cout << Contract(psi_1,psi_1.Dagger()).item() << endl;

        // new_psi = Hop->matvec(psi_1) - Bs(i-1)*psi_0;
        new_psi = Hop->matvec(psi_1);

        As.append(Contract(new_psi.Dagger(), psi_1).item().real());
        // As.append(linalg::Vectordot(new_psi.get_block_().flatten(),psi_1.get_block_().flatten(),true).item());

        unsafe_Sub_(new_psi, As(i).item(), psi_1);
        unsafe_Sub_(new_psi, Bs(i - 1).item(), psi_0);

        // diagonalize:
        tmpEsVs = linalg::Tridiag(As, Bs, true, true);

        auto tmpB =
          new_psi.Norm().item();  // sqrt(Contract(new_psi,new_psi.Dagger()).item().real());
        Bs.append(tmpB);
        if (tmpB == 0) {
          cvg_fin = true;
          break;
        }

        psi_0 = psi_1;

        new_psi /= Bs(i).item();

        psi_1 = new_psi;

        Ediff = abs(E - tmpEsVs[0].storage().at(0));
        if (verbose)
          printf("iter[%d] Enr: %11.14f, diff from last iter: %11.14f\n", i, double(E),
                 double(Ediff));

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
                          "this is intended%s",
                          "\n");
      }
      // cout << "OK" << endl;
      out.push_back(UniTensor(tmpEsVs[0](0), false, 0));

      if (is_V) {
        UniTensor eV;

        Storage kryVg = tmpEsVs[1](0).storage();
        tmpEsVs.pop_back();

        // restarted again, and evaluate the vectors on the fly:

        psi_1 = Tin.clone();
        psi_1 /= Norm;
        // psi_1.contiguous_();

        eV = kryVg.at(0) * psi_1;
        // new_psi = Hop->matvec(psi_1) - As(0)*psi_1;

        new_psi = Hop->matvec(psi_1);
        unsafe_Sub_(new_psi, As(0).item(), psi_1);

        psi_0 = psi_1;
        psi_1 = new_psi / Bs(0).item();
        for (unsigned int n = 1; n < tmpEsVs[0].shape()[0]; n++) {
          // eV += kryVg(n)*psi_1;
          unsafe_Add_(eV, kryVg.at(n), psi_1);
          // new_psi = Hop->matvec(psi_1) - Bs(n-1)*psi_0;
          new_psi = Hop->matvec(psi_1);
          unsafe_Sub_(new_psi, Bs(n - 1).item(), psi_0);
          // new_psi -= As(n)*psi_1;
          unsafe_Sub_(new_psi, As(n).item(), psi_1);

          psi_0 = psi_1;
          psi_1 = new_psi / Bs(n).item();
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

      // check Tin should be rank-1:

      UniTensor v0;
      // cytnx_error_msg(Tin.shape().size()!=1,"[ERROR][Lanczos] Tin should be rank-1%s","\n");
      // cytnx_error_msg(Tin.shape()[0]!=Hop->nx(),"[ERROR][Lanczos] Tin should have dimension
      // consistent with Hop: [%d] %s",Hop->nx(),"\n");
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

      /*
      if(Hop->dtype()==Type.ComplexDouble){
          _Lanczos_Gnd_cd(out,Hop,v0,is_V,CvgCrit,Maxiter,verbose);
      }else if(Hop->dtype()==Type.Double){
          _Lanczos_Gnd_d(out,Hop,v0,is_V,CvgCrit,Maxiter,verbose);
      }else if(Hop->dtype()==Type.ComplexFloat){
          cytnx_warning_msg(CvgCrit<1.0e-7,"[WARNING][CvgCrit] for float precision type, CvgCrit
      cannot exceed it's own type precision limit 1e-7, and it's auto capped to 1.0e-7.%s","\n");
          if(CvgCrit<1.0e-7)
              _Lanczos_Gnd_cf(out,Hop,v0,is_V,1.0e-7,Maxiter,verbose);
          else
              _Lanczos_Gnd_cf(out,Hop,v0,is_V,CvgCrit,Maxiter,verbose);
      }else{
          cytnx_warning_msg(CvgCrit<1.0e-7,"[WARNING][CvgCrit] for float precision type, CvgCrit
      cannot exceed it's own type precision limit 1e-7, and it's auto capped to 1.0e-7.%s","\n");
          if(CvgCrit<1.0e-7)
              _Lanczos_Gnd_f(out,Hop,v0,is_V,1.0e-7,Maxiter,verbose);
          else
              _Lanczos_Gnd_f(out,Hop,v0,is_V,CvgCrit,Maxiter,verbose);
      }
      */

      return out;

    }  // Lanczos_Gnd entry point

  }  // namespace linalg
}  // namespace cytnx
