#include "linalg.hpp"
#include "Generator.hpp"
#include "random.hpp"
#include "Tensor.hpp"
#include "LinOp.hpp"

#include <cfloat>
namespace cytnx {
  namespace linalg {
    typedef Accessor ac;

    void _Lanczos_ER_general(std::vector<Tensor> &out, LinOp *Hop, std::vector<Tensor> &buffer,
                             const cytnx_uint64 &k, const bool &is_V,
                             const cytnx_uint32 &max_krydim, const cytnx_uint64 &maxiter,
                             const double &CvgCrit, const bool &verbose) {
      std::vector<Tensor> converged_ev;
      // std::cout << max_krydim << std::endl;

      for (cytnx_int32 ik = 0; ik < k; ik++) {
        cytnx_uint64 krydim = max_krydim;  // initialize
        Tensor kry_mat = cytnx::zeros({krydim, krydim}, Hop->dtype(), Hop->device());

        Scalar Elast = Scalar::maxval(Type.Double);  // this is temporary

        bool cvg = false;

        // iterate start:
        for (cytnx_uint64 iter = 0; iter < maxiter; iter++) {
          // if(krydim!=kry_mat.shape()[0]) // check if krydim is dynamically adjusted.
          //     kry_mat = cytnx::zeros({krydim,krydim},Tin.dtype(),Tin.device());

          // normalized q1:
          buffer[0] = buffer[0] / buffer[0].Norm().item();  // normalized q1

          for (cytnx_uint32 ip = 1; ip < krydim + 1; ip++) {
            buffer[ip] = Hop->matvec(buffer[ip - 1]).astype(Hop->dtype());  // Hqi

            for (cytnx_uint32 ig = 0; ig < ip; ig++)
              kry_mat[{ip - 1, ig}] = Vectordot(buffer[ip], buffer[ig], true);

            // explicitly re-orthogonization
            for (cytnx_uint32 ig = 0; ig < ip; ig++) {
              buffer[ip] -= Vectordot(buffer[ig], buffer[ip], true) * buffer[ig];
              buffer[ip] /= buffer[ip].Norm().item();
            }
            // exp. reorth with previous converged ev.
            for (cytnx_uint32 ig = 0; ig < converged_ev.size(); ig++) {
              buffer[ip] -= Vectordot(converged_ev[ig], buffer[ip], true) * converged_ev[ig];
              buffer[ip] /= buffer[ip].Norm().item();
            }

          }  // ip
          auto tmp = Eigh(kry_mat, true, true);

          tmp[1] = tmp[1][{ac(0)}];  // get only the ground state.
          buffer[0] = tmp[1].get({ac(0)}) * buffer[0];
          for (cytnx_int64 ip = 1; ip < krydim; ip++)
            buffer[0] += buffer[ip] * tmp[1].get({ac(ip)});

          // check converge
          if (abs(tmp[0].storage().at(0) - Elast) < CvgCrit) {
            cvg = true;
            break;
          }
          Elast = tmp[0].storage().at(0);

        }  // for iter
        if (cvg == false) {
          cytnx_warning_msg(true,
                            "[WARNING][Lanczos] Fail to converge at eigv [%d], try increasing "
                            "maxiter?\n Note:: ignore if this is intended.%s",
                            ik, "\n");
        }
        converged_ev.push_back(buffer[0].clone());
        out[0][{ac(ik)}] = Elast;

        if (ik != k - 1) {
          /// construct a random vector for next iteration:
          while (1) {
            bool is_orth = true;
            cytnx::random::Make_normal(buffer[0], 0., 1.0);
            buffer[0] /= buffer[0].Norm().item();
            for (cytnx_uint32 ig = 0; ig < converged_ev.size(); ig++) {
              Tensor Res = Vectordot(converged_ev[ig], buffer[0], true);  // reuse variable here.
              if ((1. - Res.Norm().item()) <
                  0.005) {  // check is this vector is properly orthogonal to previous converged ev.
                is_orth = false;
                break;
              }
              buffer[0] -= Res.item() * converged_ev[ig];
              buffer[0] /= buffer[0].Norm().item();
            }
            if (is_orth) break;
          }
        }

      }  // ik

      if (is_V) {
        if (converged_ev.size() > 1) converged_ev[0].reshape_({1, -1});

        while (converged_ev.size() > 1) {
          converged_ev[0].append(converged_ev.back());
          converged_ev.pop_back();
        }
        out[1] = converged_ev[0];
      }
    }
    /*
    void _Lanczos_ER_d(std::vector<Tensor> &out, LinOp *Hop, std::vector<Tensor> &buffer, const
    cytnx_uint64 &k, const bool &is_V, const cytnx_uint32 &max_krydim, const cytnx_uint64 &maxiter,
    const double &CvgCrit, const bool &verbose){

            std::vector<Tensor> converged_ev;
            //std::cout << max_krydim << std::endl;
            for(cytnx_int32 ik=0;ik<k;ik++){
                cytnx_uint64 krydim = max_krydim; // initialize
                Tensor kry_mat = cytnx::zeros({krydim,krydim},Type.Double,Hop->device());

                double Elast = DBL_MAX; // this is temporary, so let's put it on CPU.

                bool cvg = false;

                // iterate start:
                for(cytnx_uint64 iter=0;iter<maxiter;iter++){

                    //if(krydim!=kry_mat.shape()[0]) // check if krydim is dynamically adjusted.
                    //    kry_mat = cytnx::zeros({krydim,krydim},Tin.dtype(),Tin.device());

                    // normalized q1:
                    buffer[0] /= buffer[0].Norm().item(); // normalized q1

                    for(cytnx_uint32 ip=1;ip<krydim+1;ip++){
                        buffer[ip] = Hop->matvec(buffer[ip-1]).astype(Type.Double); // Hqi

                        for(cytnx_uint32 ig=0;ig<ip;ig++)
                            kry_mat.storage().at<cytnx_double>((ip-1)*krydim+ig) =
    Vectordot(buffer[ip],buffer[ig]).item<cytnx_double>();

                        // explicitly re-orthogonization
                        for(cytnx_uint32 ig=0;ig<ip;ig++){
                            buffer[ip] -=
    Vectordot(buffer[ip],buffer[ig]).item<cytnx_double>()*buffer[ig]; buffer[ip] /=
    buffer[ip].Norm().item();
                        }
                        // exp. reorth with previous converged ev.
                        for(cytnx_uint32 ig=0;ig<converged_ev.size();ig++){
                            buffer[ip] -=
    Vectordot(buffer[ip],converged_ev[ig]).item<cytnx_double>()*converged_ev[ig]; buffer[ip] /=
    buffer[ip].Norm().item();
                        }


                    }//ip
                    auto tmp = Eigh(kry_mat,true,true);

                    tmp[1] = tmp[1](0);// get only the ground state.
                    buffer[0] = tmp[1].storage().at<cytnx_double>(0)*buffer[0];
                    for( cytnx_int64 ip=1;ip<krydim;ip++)
                        buffer[0] +=
    buffer[ip]*tmp[1].storage().at<cytnx_double>(ip);//get({ac(ip)});


                    // check converge
                    if(verbose){
                        printf("iter[%d] Enr: %11.14f, diff from last iter:
    %11.14f\n",iter,tmp[0].storage().at<cytnx_double>(0),std::abs(tmp[0].storage().at<cytnx_double>(0)
    - Elast));
                    }
                    if(std::abs(tmp[0].storage().at<cytnx_double>(0) - Elast) < CvgCrit){
                        cvg = true;
                        break;
                    }
                    Elast = tmp[0].storage().at<cytnx_double>(0);

                }//for iter
                if(cvg==false){
                    cytnx_warning_msg(true,"[WARNING][Lanczos] Fail to converge at eigv [%d], try
    increasing maxiter?\n Note:: ignore if this is intended.%s",ik,"\n");
                }
                converged_ev.push_back(buffer[0].clone());
                out[0].storage().at<cytnx_double>(ik) = Elast;


                if(ik!=k-1){
                    ///construct a random vector for next iteration:
                    while(1){
                        bool is_orth=true;
                        cytnx::random::Make_normal(buffer[0],0.,1.0);
                        buffer[0]/=buffer[0].Norm().item();
                        for(cytnx_uint32 ig=0;ig<converged_ev.size();ig++){
                            Elast = Vectordot(buffer[0],converged_ev[ig]).item<cytnx_double>();
    //reuse variable here. if((1-Elast) < 0.005){ // check is this vector is properly orthogonal to
    previous converged ev. is_orth=false; break;
                            }
                            buffer[0] -= Elast*converged_ev[ig];
                            buffer[0] /= buffer[0].Norm().item();
                        }
                        if(is_orth)
                            break;
                    }
                }

            }// ik

            if(is_V){
                if(converged_ev.size()>1)
                    converged_ev[0].reshape_({1,-1});

                while(converged_ev.size()>1){
                    converged_ev[0].append(converged_ev.back());
                    converged_ev.pop_back();
                }
                out[1] = converged_ev[0];
            }
    }



    void _Lanczos_ER_f(std::vector<Tensor> &out, LinOp *Hop, std::vector<Tensor> &buffer, const
    cytnx_uint64 &k, const bool &is_V, const cytnx_uint32 &max_krydim, const cytnx_uint64 &maxiter,
    const double &CvgCrit, const bool &verbose){

            std::vector<Tensor> converged_ev;
            //std::cout << max_krydim << std::endl;
            for(cytnx_int32 ik=0;ik<k;ik++){
                cytnx_uint64 krydim = max_krydim; // initialize
                Tensor kry_mat = cytnx::zeros({krydim,krydim},Type.Float,Hop->device());

                float Elast = FLT_MAX; // this is temporary, so let's put it on CPU.

                bool cvg = false;

                // iterate start:
                for(cytnx_uint64 iter=0;iter<maxiter;iter++){

                    //if(krydim!=kry_mat.shape()[0]) // check if krydim is dynamically adjusted.
                    //    kry_mat = cytnx::zeros({krydim,krydim},Tin.dtype(),Tin.device());

                    // normalized q1:
                    buffer[0] = buffer[0]/buffer[0].Norm().item(); // normalized q1

                    for(cytnx_uint32 ip=1;ip<krydim+1;ip++){
                        buffer[ip] = Hop->matvec(buffer[ip-1]).astype(Type.Float); // Hqi

                        for(cytnx_uint32 ig=0;ig<ip;ig++)
                            kry_mat[{ip-1,ig}] = Vectordot(buffer[ip],buffer[ig]);

                        // explicitly re-orthogonization
                        for(cytnx_uint32 ig=0;ig<ip;ig++){
                            buffer[ip] -= Vectordot(buffer[ip],buffer[ig])*buffer[ig];
                            buffer[ip] /= buffer[ip].Norm().item();
                        }
                        // exp. reorth with previous converged ev.
                        for(cytnx_uint32 ig=0;ig<converged_ev.size();ig++){
                            buffer[ip] -= Vectordot(buffer[ip],converged_ev[ig])*converged_ev[ig];
                            buffer[ip] /= buffer[ip].Norm().item();
                        }


                    }//ip
                    auto tmp = Eigh(kry_mat,true,true);

                    tmp[1] = tmp[1][{ac(0)}];// get only the ground state.
                    buffer[0] = tmp[1].get({ac(0)})*buffer[0];
                    for( cytnx_int64 ip=1;ip<krydim;ip++)
                        buffer[0] += buffer[ip]*tmp[1].get({ac(ip)});

                    // check converge
                    if(std::abs(tmp[0].storage().at<cytnx_float>(0) - Elast) < CvgCrit){
                        cvg = true;
                        break;
                    }
                    Elast = tmp[0].storage().at<cytnx_float>(0);

                }//for iter
                if(cvg==false){
                    cytnx_warning_msg(true,"[WARNING][Lanczos] Fail to converge at eigv [%d], try
    increasing maxiter?\n Note:: ignore if this is intended.%s",ik,"\n");
                }
                converged_ev.push_back(buffer[0].clone());
                out[0][{ac(ik)}] = Elast;


                if(ik!=k-1){
                    ///construct a random vector for next iteration:
                    while(1){
                        bool is_orth=true;
                        cytnx::random::Make_normal(buffer[0],0.,1.0);
                        buffer[0]/=buffer[0].Norm().item();
                        for(cytnx_uint32 ig=0;ig<converged_ev.size();ig++){
                            Elast = Vectordot(buffer[0],converged_ev[ig]).item<cytnx_float>();
    //reuse variable here. if((1-Elast) < 0.005){ // check is this vector is properly orthogonal to
    previous converged ev. is_orth=false; break;
                            }
                            buffer[0] -= Elast*converged_ev[ig];
                            buffer[0] /= buffer[0].Norm().item();
                        }
                        if(is_orth)
                            break;
                    }
                }

            }// ik

            if(is_V){
                if(converged_ev.size()>1)
                    converged_ev[0].reshape_({1,-1});

                while(converged_ev.size()>1){
                    converged_ev[0].append(converged_ev.back());
                    converged_ev.pop_back();
                }
                out[1] = converged_ev[0];
            }
    }

    void _Lanczos_ER_cf(std::vector<Tensor> &out, LinOp *Hop, std::vector<Tensor> &buffer, const
    cytnx_uint64 &k, const bool &is_V, const cytnx_uint32 &max_krydim, const cytnx_uint64 &maxiter,
    const double &CvgCrit, const bool &verbose){

            std::vector<Tensor> converged_ev;
            //std::cout << max_krydim << std::endl;
            for(cytnx_int32 ik=0;ik<k;ik++){
                cytnx_uint64 krydim = max_krydim; // initialize
                Tensor kry_mat = cytnx::zeros({krydim,krydim},Type.ComplexFloat,Hop->device());

                float Elast = FLT_MAX; // this is temporary, so let's put it on CPU.

                bool cvg = false;

                // iterate start:
                for(cytnx_uint64 iter=0;iter<maxiter;iter++){

                    //if(krydim!=kry_mat.shape()[0]) // check if krydim is dynamically adjusted.
                    //    kry_mat = cytnx::zeros({krydim,krydim},Tin.dtype(),Tin.device());

                    // normalized q1:
                    buffer[0] = buffer[0]/buffer[0].Norm().item(); // normalized q1

                    for(cytnx_uint32 ip=1;ip<krydim+1;ip++){
                        buffer[ip] = Hop->matvec(buffer[ip-1]).astype(Type.ComplexFloat); // Hqi

                        for(cytnx_uint32 ig=0;ig<ip;ig++)
                            kry_mat[{ip-1,ig}] = Vectordot(buffer[ip],buffer[ig],true);

                        // explicitly re-orthogonization
                        for(cytnx_uint32 ig=0;ig<ip;ig++){
                            buffer[ip] -= Vectordot(buffer[ig],buffer[ip],true)*buffer[ig];
                            buffer[ip] /= buffer[ip].Norm().item();
                        }
                        // exp. reorth with previous converged ev.
                        for(cytnx_uint32 ig=0;ig<converged_ev.size();ig++){
                            buffer[ip] -=
    Vectordot(converged_ev[ig],buffer[ip],true)*converged_ev[ig]; buffer[ip] /=
    buffer[ip].Norm().item();
                        }


                    }//ip
                    auto tmp = Eigh(kry_mat,true,true);

                    tmp[1] = tmp[1][{ac(0)}];// get only the ground state.
                    buffer[0] = tmp[1].get({ac(0)})*buffer[0];
                    for( cytnx_int64 ip=1;ip<krydim;ip++)
                        buffer[0] += buffer[ip]*tmp[1].get({ac(ip)});

                    // check converge
                    if(std::abs(tmp[0].storage().at<cytnx_float>(0) - Elast) < CvgCrit){
                        cvg = true;
                        break;
                    }
                    Elast = tmp[0].storage().at<cytnx_float>(0);

                }//for iter
                if(cvg==false){
                    cytnx_warning_msg(true,"[WARNING][Lanczos] Fail to converge at eigv [%d], try
    increasing maxiter?\n Note:: ignore if this is intended.%s",ik,"\n");
                }
                converged_ev.push_back(buffer[0].clone());
                out[0][{ac(ik)}] = Elast;


                if(ik!=k-1){
                    ///construct a random vector for next iteration:
                    while(1){
                        bool is_orth=true;
                        cytnx::random::Make_normal(buffer[0],0.,1.0);
                        buffer[0]/=buffer[0].Norm().item();
                        for(cytnx_uint32 ig=0;ig<converged_ev.size();ig++){
                            Tensor Res = Vectordot(converged_ev[ig],buffer[0],true); //reuse
    variable here. if((1-Res.Norm().item()) < 0.005){ // check is this vector is properly orthogonal
    to previous converged ev. is_orth=false; break;
                            }
                            buffer[0] -= Res.item<cytnx_complex64>()*converged_ev[ig];
                            buffer[0] /= buffer[0].Norm().item();
                        }
                        if(is_orth)
                            break;
                    }
                }

            }// ik

            if(is_V){
                if(converged_ev.size()>1)
                    converged_ev[0].reshape_({1,-1});

                while(converged_ev.size()>1){
                    converged_ev[0].append(converged_ev.back());
                    converged_ev.pop_back();
                }
                out[1] = converged_ev[0];
            }
    }

    void _Lanczos_ER_cd(std::vector<Tensor> &out, LinOp *Hop, std::vector<Tensor> &buffer, const
    cytnx_uint64 &k, const bool &is_V, const cytnx_uint32 &max_krydim, const cytnx_uint64 &maxiter,
    const double &CvgCrit, const bool &verbose){

            std::vector<Tensor> converged_ev;
            //std::cout << max_krydim << std::endl;
            for(cytnx_int32 ik=0;ik<k;ik++){
                cytnx_uint64 krydim = max_krydim; // initialize
                Tensor kry_mat = cytnx::zeros({krydim,krydim},Type.ComplexDouble,Hop->device());
                //std::cout << "cool" << std::endl;
                double Elast = DBL_MAX; // this is temporary, so let's put it on CPU.

                bool cvg = false;

                // iterate start:
                for(cytnx_uint64 iter=0;iter<maxiter;iter++){

                    //if(krydim!=kry_mat.shape()[0]) // check if krydim is dynamically adjusted.
                    //    kry_mat = cytnx::zeros({krydim,krydim},Tin.dtype(),Tin.device());

                    // normalized q1:
                    buffer[0] = buffer[0]/buffer[0].Norm().item(); // normalized q1

                    for(cytnx_uint32 ip=1;ip<krydim+1;ip++){
                        buffer[ip] = Hop->matvec(buffer[ip-1]).astype(Type.ComplexDouble); // Hqi

                        for(cytnx_uint32 ig=0;ig<ip;ig++)
                            kry_mat[{ip-1,ig}] = Vectordot(buffer[ip],buffer[ig],true);

                        // explicitly re-orthogonization
                        for(cytnx_uint32 ig=0;ig<ip;ig++){
                            buffer[ip] -= Vectordot(buffer[ig],buffer[ip],true)*buffer[ig];
                            buffer[ip] /= buffer[ip].Norm().item();
                        }
                        // exp. reorth with previous converged ev.
                        for(cytnx_uint32 ig=0;ig<converged_ev.size();ig++){
                            buffer[ip] -=
    Vectordot(converged_ev[ig],buffer[ip],true)*converged_ev[ig]; buffer[ip] /=
    buffer[ip].Norm().item();
                        }


                    }//ip
                    auto tmp = Eigh(kry_mat,true,true);

                    tmp[1] = tmp[1][{ac(0)}];// get only the ground state.
                    buffer[0] = tmp[1].get({ac(0)})*buffer[0];
                    for( cytnx_int64 ip=1;ip<krydim;ip++)
                        buffer[0] += buffer[ip]*tmp[1].get({ac(ip)});

                    // check converge
                    if(std::abs(tmp[0].storage().at<cytnx_double>(0) - Elast) < CvgCrit){
                        cvg = true;
                        break;
                    }
                    Elast = tmp[0].storage().at<cytnx_double>(0);

                }//for iter
                if(cvg==false){
                    cytnx_warning_msg(true,"[WARNING][Lanczos] Fail to converge at eigv [%d], try
    increasing maxiter?\n Note:: ignore if this is intended.%s",ik,"\n");
                }
                converged_ev.push_back(buffer[0].clone());
                out[0][{ac(ik)}] = Elast;


                if(ik!=k-1){
                    ///construct a random vector for next iteration:
                    while(1){
                        bool is_orth=true;
                        cytnx::random::Make_normal(buffer[0],0.,1.0);
                        buffer[0]/=buffer[0].Norm().item();
                        for(cytnx_uint32 ig=0;ig<converged_ev.size();ig++){
                            Tensor Res = Vectordot(converged_ev[ig],buffer[0],true); //reuse
    variable here. if((1.-Res.Norm().item()) < 0.005){ // check is this vector is properly
    orthogonal to previous converged ev. is_orth=false; break;
                            }
                            buffer[0] -= Res.item<cytnx_complex128>()*converged_ev[ig];
                            buffer[0] /= buffer[0].Norm().item();
                        }
                        if(is_orth)
                            break;
                    }
                }

            }// ik

            if(is_V){
                if(converged_ev.size()>1)
                    converged_ev[0].reshape_({1,-1});

                while(converged_ev.size()>1){
                    converged_ev[0].append(converged_ev.back());
                    converged_ev.pop_back();
                }
                out[1] = converged_ev[0];
            }
    }
    */

    // MERL
    // https://www.sciencedirect.com/science/article/pii/S0010465597001367

    // explicitly re-started Lanczos
    std::vector<Tensor> Lanczos_ER(LinOp *Hop, const cytnx_uint64 &k, const bool &is_V,
                                   const cytnx_uint64 &maxiter, const double &CvgCrit,
                                   const bool &is_row, const Tensor &Tin,
                                   const cytnx_uint32 &max_krydim, const bool &verbose) {
      // check type:
      cytnx_error_msg(
        !Type.is_float(Hop->dtype()),
        "[ERROR][Lanczos] Lanczos can only accept operator with floating types (complex/real)%s",
        "\n");

      // krydim we start from 3, and increase to max_krydim. So max_krydim must >=3
      cytnx_error_msg(max_krydim < 2, "[ERROR][Lanczos] max_krydim must >=2%s", "\n");

      /// check k
      cytnx_error_msg(k < 1, "[ERROR][Lanczos] k should be >0%s", "\n");
      cytnx_error_msg(k >= Hop->nx(),
                      "[ERROR][Lanczos] k can only be up to total dimension of input vector D-1%s",
                      "\n");

      // check Tin should be rank-1:
      std::vector<Tensor> buffer(max_krydim + 1);
      if (Tin.dtype() == Type.Void) {
        buffer[0] =
          cytnx::random::normal({Hop->nx()}, Hop->dtype(), Hop->device());  // randomly initialize.
      } else {
        cytnx_error_msg(Tin.shape().size() != 1, "[ERROR][Lanczos] Tin should be rank-1%s", "\n");
        cytnx_error_msg(Tin.shape()[0] != Hop->nx(),
                        "[ERROR][Lanczos] Tin should have dimension consistent with Hop: [%d] %s",
                        Hop->nx(), "\n");
        buffer[0] = Tin.astype(Hop->dtype());
      }

      // std::cout << "[entry] LER" << std::endl;

      std::vector<Tensor> out;
      if (is_V)
        out.resize(2);
      else
        out.resize(1);

      out[0] =
        zeros({k}, Type.is_complex(Hop->dtype()) ? Hop->dtype() + 2 : Hop->dtype(), Hop->device());
      // std::cout << "[entry] LER 2" << std::endl;

      double _cvgcrit = CvgCrit;
      if (Hop->dtype() == Type.Float || Hop->dtype() == Type.ComplexFloat) {
        if (_cvgcrit < 1.0e-7) {
          _cvgcrit = 1.0e-7;
          cytnx_warning_msg(CvgCrit < 1.0e-7,
                            "[ERROR][CvgCrit] for float precision type, CvgCrit cannot exceed it's "
                            "own type precision limit 1e-7, and it's auto capped to 1.0e-7.%s",
                            "\n");
        }
      }

      _Lanczos_ER_general(out, Hop, buffer, k, is_V, max_krydim, maxiter, CvgCrit, verbose);

      /*
      if(Hop->dtype()==Type.ComplexDouble){
          _Lanczos_ER_cd(out,Hop, buffer,k,is_V,max_krydim,maxiter,CvgCrit,verbose);
      }else if(Hop->dtype()==Type.Double){
          _Lanczos_ER_d(out,Hop, buffer,k,is_V,max_krydim,maxiter,CvgCrit,verbose);
      }else if(Hop->dtype()==Type.ComplexFloat){
          cytnx_warning_msg(CvgCrit<1.0e-7,"[ERROR][CvgCrit] for float precision type, CvgCrit
      cannot exceed it's own type precision limit 1e-7, and it's auto capped to 1.0e-7.%s","\n");
          if(CvgCrit<1.0e-7)
              _Lanczos_ER_cf(out,Hop, buffer,k,is_V,max_krydim,maxiter,1.0e-7,verbose);
          else
              _Lanczos_ER_cf(out,Hop, buffer,k,is_V,max_krydim,maxiter,CvgCrit,verbose);
      }else{
          cytnx_warning_msg(CvgCrit<1.0e-7,"[ERROR][CvgCrit] for float precision type, CvgCrit
      cannot exceed it's own type precision limit 1e-7, and it's auto capped to 1.0e-7.%s","\n");
          if(CvgCrit<1.0e-7)
              _Lanczos_ER_f(out,Hop, buffer,k,is_V,max_krydim,maxiter,1.0e-7,verbose);
          else
              _Lanczos_ER_f(out,Hop, buffer,k,is_V,max_krydim,maxiter,CvgCrit,verbose);
      }
      */

      if (!is_row) {
        out[1].Conj_();
        if (out[1].shape().size() != 1) out[1].permute_({1, 0});
      }

      return out;

    }  // Lanczos_ER entry point

  }  // namespace linalg
}  // namespace cytnx
