#include "linalg.hpp"
#include "Generator.hpp"
#include "random.hpp"
#include "Tensor.hpp"
#include "LinOp.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>
#include "Tensor.hpp"
#include <iomanip>

#include "Lanczos_Gnd_breakdown.hpp"

namespace cytnx {
  namespace linalg {
    typedef Accessor ac;

    void _Lanczos_Gnd_general(std::vector<Tensor> &out, LinOp *Hop, const Tensor &Tin,
                              const bool &is_V, const double &CvgCrit, const unsigned int &Maxiter,
                              const bool &verbose) {
      out.clear();
      //[require] Tin should be provided!

      const double tin_norm = double(Tin.norm());
      cytnx_error_msg(!(tin_norm > 0), "[ERROR][Lanczos] the initial vector Tin must be nonzero.%s",
                      "\n");

      // The Lanczos coefficients of a Hermitian operator are real.
      const unsigned int coeff_dtype = Type.to_real(Hop->dtype());
      const double breakdown_tol = lanczos_gnd_breakdown_tolerance(coeff_dtype);
      // At most dim linearly independent Krylov vectors exist.
      const cytnx_uint64 dim = Tin.shape()[0];

      Tensor psi_1 = Tin.clone() / tin_norm;

      Tensor psi_0;  // = cytnx::zeros({psi_1.shape()[0]},psi_1.dtype(),Tin.device());
      Tensor new_psi;
      bool cvg_fin = false;

      // declare variables:
      Tensor As = zeros({1}, coeff_dtype, Tin.device());
      Tensor Bs = As.clone();

      // temporary:
      std::vector<Tensor> tmpEsVs;

      // i=0
      //-------------------------------------------------
      new_psi = Hop->matvec(psi_1);

      Scalar alpha = linalg::Vectordot(new_psi, psi_1, true).item().real();
      As(0) = alpha;
      new_psi -= As(0) * psi_1;

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
        new_psi /= Bs(0);

        psi_1 = new_psi;

        Scalar E = alpha;
        Scalar Ediff;

        ///---------------------------

        const unsigned int max_krylov_dim =
          static_cast<unsigned int>(std::min<cytnx_uint64>(Maxiter, dim));

        // iteration LZ:
        for (unsigned int i = 1; i < max_krylov_dim; i++) {
          new_psi = Hop->matvec(psi_1);  //- Bs(i-1)*psi_0;

          alpha = linalg::Vectordot(new_psi, psi_1, true).item().real();
          As.append(alpha);

          new_psi -= As(i) * psi_1;
          new_psi -= Bs(i - 1) * psi_0;

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
          new_psi /= Bs(i);

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
      }

      if (cvg_fin == false) {
        cytnx_warning_msg(true,
                          "[WARNING] iteration not converge after Maxiter!.\n :: Note :: ignore if "
                          "this is intended%s",
                          "\n");
      }

      out.push_back(tmpEsVs[0](0));

      if (is_V) {
        Tensor eV;
        Tensor kryVg = tmpEsVs[1](0);
        tmpEsVs.pop_back();

        // restarted again, and evaluate the vectors on the fly:
        psi_1 = Tin.clone() / tin_norm;
        eV = kryVg(0) * psi_1;

        for (unsigned int n = 1; n < tmpEsVs[0].shape()[0]; n++) {
          if (n == 1) {
            new_psi = Hop->matvec(psi_1) - As(0) * psi_1;
          } else {
            new_psi = Hop->matvec(psi_1) - Bs(n - 2) * psi_0;
            new_psi -= As(n - 1) * psi_1;
          }

          psi_0 = psi_1;
          psi_1 = new_psi / Bs(n - 1);
          eV += kryVg(n) * psi_1;
        }

        out.push_back(eV);
      }
    }
    /*
    void _Lanczos_Gnd_d(std::vector<Tensor> &out, LinOp *Hop, const Tensor &Tin, const bool &is_V,
    const double &CvgCrit, const unsigned int &Maxiter, const bool &verbose){ out.clear();
        //[require] Tin should be provided!


        Tensor psi_1 = Tin.clone()/Tin.Norm();

        Tensor psi_0;// = cytnx::zeros({psi_1.shape()[0]},psi_1.dtype(),Tin.device());
        Tensor new_psi;
        bool cvg_fin=false;

        //declare variables:
        Tensor As = zeros({1},Tin.dtype(),Tin.device());
        Tensor Bs = As.clone();

        double E=DBL_MAX;


        //temporary:
        std::vector<Tensor> tmpEsVs;


        // i=0
        //-------------------------------------------------
        new_psi = Hop->matvec(psi_1);
        As(0) = linalg::Vectordot(new_psi,psi_1).item<double>();
        new_psi -= As(0)*psi_1;

        Bs(0) = new_psi.Norm();
        psi_0 = psi_1;
        new_psi /= Bs(0);


        psi_1 = new_psi;

        E = As(0).item<double>();
        double Ediff;

        ///---------------------------


        //iteration LZ:
        for(unsigned int i=1;i<Maxiter;i++){

            new_psi = Hop->matvec(psi_1) - Bs(i-1)*psi_0;
            As.append(linalg::Vectordot(new_psi,psi_1).item<double>());
            new_psi -= As(i)*psi_1;

            //diagonalize:
            tmpEsVs = linalg::Tridiag(As,Bs,true,true);


            Bs.append(new_psi.Norm().item<double>());
            psi_0 = psi_1;
            new_psi /= Bs(i);

            psi_1 = new_psi;

            Ediff = std::abs(E - tmpEsVs[0].storage().at<double>(0));
            if(verbose) printf("iter[%d] Enr: %11.14f, diff from last iter: %11.14f\n",i,E,Ediff);

            //chkf = true;
            if(Ediff < CvgCrit){
                cvg_fin=true;
                break;
            }
            E = tmpEsVs[0].storage().at<double>(0);

        }//iteration

        if(cvg_fin==false){
            cytnx_warning_msg(true,"[WARNING] iteration not converge after Maxiter!.\n :: Note ::
    ignore if this is intended","\n");
        }


        out.push_back(tmpEsVs[0](0));

        if(is_V){
            Tensor eV;
            Tensor kryVg = tmpEsVs[1](0);
            tmpEsVs.pop_back();

            //restarted again, and evaluate the vectors on the fly:
            psi_1 = Tin.clone()/Tin.Norm();
            eV = kryVg(0)*psi_1;
            new_psi = Hop->matvec(psi_1) - As(0)*psi_1;

            psi_0 = psi_1;
            psi_1 = new_psi/Bs(0);

            for(unsigned int n=1;n<tmpEsVs[0].shape()[0];n++){
                eV += kryVg(n)*psi_1;
                new_psi = Hop->matvec(psi_1) - Bs(n-1)*psi_0;
                new_psi -= As(n)*psi_1;

                psi_0 = psi_1;
                psi_1 = new_psi/Bs(n);

            }

            out.push_back(eV);
        }





    }
    void _Lanczos_Gnd_f(std::vector<Tensor> &out, LinOp *Hop, const Tensor &Tin, const bool &is_V,
    const double &CvgCrit, const unsigned int &Maxiter, const bool &verbose){ out.clear();
        //[require] Tin should be provided!


        Tensor psi_1 = Tin.clone();
        psi_1 /= psi_1.Norm();

        Tensor psi_0;// = cytnx::zeros({psi_1.shape()[0]},psi_1.dtype(),Tin.device());
        Tensor new_psi;
        bool cvg_fin=false;

        //declare variables:
        Tensor As = zeros({1},Tin.dtype(),Tin.device());
        Tensor Bs = As.clone();

        float E=std::numeric_limits<float>::max();


        //temporary:
        std::vector<Tensor> tmpEsVs;


        // i=0
        //-------------------------------------------------
        new_psi = Hop->matvec(psi_1).astype(Type.Float);
        As(0) = linalg::Vectordot(new_psi,psi_1).item();
        new_psi -= As(0)*psi_1;

        Bs(0) = new_psi.Norm();
        psi_0 = psi_1;
        new_psi /= Bs(0);


        psi_1 = new_psi;

        E = As(0).item<float>();
        float Ediff;

        ///---------------------------


        //iteration LZ:
        for(unsigned int i=1;i<Maxiter;i++){

            new_psi = Hop->matvec(psi_1).astype(Type.Float) - Bs(i-1)*psi_0;
            As.append(linalg::Vectordot(new_psi,psi_1).item<float>());
            new_psi -= As(i)*psi_1;

            //diagonalize:
            tmpEsVs = linalg::Tridiag(As,Bs,true,true);


            Bs.append(float(Scalar(new_psi.Norm().item())));
            psi_0 = psi_1;
            new_psi /= Bs(i);

            psi_1 = new_psi;

            Ediff = std::abs(E - tmpEsVs[0].storage().at<float>(0));
            if(verbose) printf("iter[%d] Enr: %11.14f, diff from last iter: %11.14f\n",i,E,Ediff);

            //chkf = true;
            if(Ediff < CvgCrit){
                cvg_fin=true;
                break;
            }
            E = tmpEsVs[0].storage().at<float>(0);

        }//iteration

        if(cvg_fin==false){
            cytnx_warning_msg(true,"[WARNING] iteration not converge after Maxiter!.\n :: Note ::
    ignore if this is intended","\n");
        }


        out.push_back(tmpEsVs[0](0));

        if(is_V){
            Tensor eV;
            Tensor kryVg = tmpEsVs[1](0);
            tmpEsVs.pop_back();

            //restarted again, and evaluate the vectors on the fly:
            psi_1 = Tin.clone();
            psi_1/=Tin.Norm().item();
            eV = kryVg(0)*psi_1;
            new_psi = Hop->matvec(psi_1) - As(0)*psi_1;

            psi_0 = psi_1;
            psi_1 = new_psi/Bs(0);

            for(unsigned int n=1;n<tmpEsVs[0].shape()[0];n++){
                eV += kryVg(n)*psi_1;
                new_psi = Hop->matvec(psi_1).astype(Type.Float) - Bs(n-1)*psi_0;
                new_psi -= As(n)*psi_1;

                psi_0 = psi_1;
                psi_1 = new_psi/Bs(n);

            }

            out.push_back(eV);
        }


    }
    void _Lanczos_Gnd_cf(std::vector<Tensor> &out, LinOp *Hop, const Tensor &Tin, const bool &is_V,
    const double &CvgCrit, const unsigned int &Maxiter, const bool &verbose){
        cytnx_error_msg(true,"[Developing]%s","\n");
    }
    void _Lanczos_Gnd_cd(std::vector<Tensor> &out, LinOp *Hop, const Tensor &Tin, const bool &is_V,
    const double &CvgCrit, const unsigned int &Maxiter, const bool &verbose){
        cytnx_error_msg(true,"[Developing]%s","\n");
    }
    */

    // Lanczos
    std::vector<Tensor> Lanczos_Gnd(LinOp *Hop, const double &CvgCrit, const bool &is_V,
                                    const Tensor &Tin, const bool &verbose,
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
      Tensor v0;
      if (Tin.dtype() == Type.Void) {
        v0 = cytnx::random::normal({Hop->nx()}, 0, 1, Hop->device())
               .astype(Hop->dtype());  // randomly initialize.
      } else {
        cytnx_error_msg(Tin.shape().size() != 1, "[ERROR][Lanczos] Tin should be rank-1%s", "\n");
        cytnx_error_msg(Tin.shape()[0] != Hop->nx(),
                        "[ERROR][Lanczos] Tin should have dimension consistent with Hop: [%d] %s",
                        Hop->nx(), "\n");
        v0 = Tin.astype(Hop->dtype());
      }

      std::vector<Tensor> out;

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

      _Lanczos_Gnd_general(out, Hop, v0, is_V, _cvgcrit, Maxiter, verbose);

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
