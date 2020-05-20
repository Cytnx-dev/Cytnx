#include "linalg.hpp"
#include "linalg_internal_interface.hpp"

#include <iostream>
#include <vector>

using namespace std;
typedef cytnx::Accessor ac;
namespace cytnx{
    namespace linalg{
         
        
        std::vector<Tensor> QR(const Tensor &Tin, const bool &is_tau){
            
            cytnx_error_msg(Tin.shape().size() != 2,"[QR] error, QR can only operate on rank-2 Tensor.%s","\n");
            cytnx_error_msg(!Tin.is_contiguous(), "[QR] error tensor must be contiguous. Call Contiguous_() or Contiguous() first%s","\n");
            
            
            cytnx_uint64 n_tau = std::max(cytnx_uint64(1),std::min(Tin.shape()[0],Tin.shape()[1])); 

            Tensor in;
            if(Tin.dtype() > Type.Float) in = Tin.astype(Type.Double);
            else in = Tin;

            
            

            //std::cout << n_singlu << std::endl;

            Tensor tau,Q,R;
            tau.Init({n_tau},in.dtype(),in.device()); tau.storage().set_zeros();// if type is complex, S should be real
            Q.Init(Tin.shape(),in.dtype(),in.device()); Q.storage().set_zeros();
            R.Init({n_tau,Tin.shape()[1]},in.dtype(),in.device()); R.storage().set_zeros();

            if(Tin.device()==Device.cpu){


                cytnx::linalg_internal::lii.QR_ii[in.dtype()](in._impl->storage()._impl, 
                                                        Q._impl->storage()._impl,
                                                        R._impl->storage()._impl,  
                                                        tau._impl->storage()._impl,in.shape()[0],in.shape()[1]);

                std::vector<Tensor> out;
                if(in.shape()[0]<in.shape()[1]) Q = Q[{ac::all(),ac::range(0,in.shape()[0],1)}]; 
                out.push_back(Q);
                out.push_back(R);
                
                if(is_tau) out.push_back(tau);
                
                return out;

            }else{
                #ifdef UNI_GPU
                    cytnx_error_msg(true,"[QR] error,%s","Currently QR does not support CUDA.\n");
                    /*
                    checkCudaErrors(cudaSetDevice(in.device()));
                    cytnx::linalg_internal::lii.cuQR_ii[in.dtype()](in._impl->storage()._impl,
                                                            U._impl->storage()._impl,
                                                            vT._impl->storage()._impl,
                                                            S._impl->storage()._impl,in.shape()[0],in.shape()[1]);

                    std::vector<Tensor> out;
                    out.push_back(S);
                    if(is_U) out.push_back(U);
                    if(is_vT) out.push_back(vT);
                    
                    return out;
                    */
                    return std::vector<Tensor>();
                #else
                    cytnx_error_msg(true,"[QR] fatal error,%s","try to call the gpu section without CUDA support.\n");
                    return std::vector<Tensor>();
                #endif
            }    

        }




    }//linalg namespace

}//cytnx namespace



