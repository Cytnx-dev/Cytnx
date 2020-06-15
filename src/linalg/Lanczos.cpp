#include "linalg.hpp"
#include "Generator.hpp"
#include "random.hpp"
namespace cytnx{
    namespace linalg{
        typedef Accessor ac;

        

        void _Lanczos_ER_cd(std::vector<Tensor> &out, LinOp *Hop, const Tensor &Tin, const cytnx_uint64 &k, const bool &is_V, const cytnx_uint32 &max_krydim, const cytnx_uint64 &maxiter, const double &CvgCrit, const bool &is_row){
                std::vector<Tensor> buffer(max_krydim+1);
                buffer[0] = Tin;

                for(cytnx_int32 ik=0;ik<k;ik++){
                    cytnx_uint64 krydim = max_krydim; // initialize
                    Tensor kry_mat = cytnx::zeros({krydim,krydim},Type.Double,Tin.device());

                    double Elast = 0; // this is temporary, so let's put it on CPU.    

                    bool cvg = false;

                    // iterate start:
                    for(cytnx_uint64 iter=0;iter<maxiter;iter++){
                    
                        //if(krydim!=kry_mat.shape()[0]) // check if krydim is dynamically adjusted.
                        //    kry_mat = cytnx::zeros({krydim,krydim},Tin.dtype(),Tin.device());
            
                        // normalized q1:
                        buffer[0] = buffer[0]/buffer[0].Norm().item<cytnx_double>(); // normalized q1

                        for(cytnx_uint32 ip=1;ip<krydim;ip++){
                            buffer[ip] = Hop->matvec(buffer[ip-1]); // Hqi
                            
                            for(cytnx_uint32 ig=0;ig<ip;ig++)
                                kry_mat[{ip-1,ig}] = Vectordot(buffer[ip],buffer[ig],true).real();
                            
                            // explicitly re-orthogonization
                            for(cytnx_uint32 ig=0;ig<ip;ig++){
                                buffer[ip] -= Vectordot(buffer[ip],buffer[ig],true)*buffer[ig];
                                buffer[ip] /= buffer[ip].Norm().item<cytnx_double>();
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
                        cytnx_error_msg(true,"[ERROR][Lanczos] Fail to converge at eigv [%d], try increasing maxiter?%s",ik,"\n");
                    }
                    out[1].set({ac(ik)},buffer[0]);     
                    out[0][{ac(ik)}] = Elast;
                    cytnx::random::Make_normal(buffer[0],0.,1.0);
                }


        }



        // MERL
        // https://www.sciencedirect.com/science/article/pii/S0010465597001367               

        // explicitly re-started Lanczos        
        std::vector<Tensor> Lanczos_ER(LinOp *Hop, const Tensor &Tin, const cytnx_uint64 &k, const bool &is_V, const cytnx_uint32 &max_krydim, const cytnx_uint64 &maxiter, const double &CvgCrit, const bool &is_row){
                // krydim we start from 3, and increase to max_krydim. So max_krydim must >=3
                cytnx_error_msg(max_krydim<3,"[ERROR][Lanczos] max_krydim must >=3%s","\n");
            
                // check Tin should be rank-1 + contiguous:
                cytnx_error_msg(Tin.shape().size()!=1,"[ERROR][Lanczos] Tin should be rank-1%s","\n");
                cytnx_error_msg(!Tin.is_contiguous(),"[ERROR][Lanczos] Tin should be contiguous. Call .contiguous() or .contiguous_() first %s","\n");
                
                /// check k
                cytnx_error_msg(k<1,"[ERROR][Lanczos] k should be >0%s","\n");

                // check type of Tin: 
                cytnx_error_msg(!Type.is_float(Tin.dtype()),"[ERROR][Lanczos] can only accept Tin to be real/complex floating type (double/float).%s","\n");


                std::vector<Tensor> out;
                if(is_V) out.resize(2);
                else     out.resize(1);
                
                out[0] = zeros({k},Type.is_complex(Tin.dtype())?Tin.dtype()-2:Tin.dtype(),Tin.device());
                if(is_V) out[1] = cytnx::zeros({k,Tin.shape()[0]},Tin.dtype(),Tin.device());
                
                cytnx_error_msg(true, "[ERROR][Developing.]%s","\n");                

                
                return out;

        }///Lanczos

        


    }//linalg
}// cytnx

