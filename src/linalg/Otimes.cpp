#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "utils/utils.hpp"
#include "Device.hpp"
#include <iostream>
namespace cytnx{

    namespace linalg{
        Tensor Otimes(const Tensor &Tl, const Tensor &Tr, const bool &matrix_form){
            
            //checking:
            cytnx_error_msg(Tl.shape().size()==0,"[ERROR] pass empty tensor in param #1%s","\n");
            cytnx_error_msg(Tr.shape().size()==0,"[ERROR] pass empty tensor in param #2%s","\n");
            cytnx_error_msg(Tl.device()!= Tr.device(),"[ERROR] two tensor cannot on different devices.%s","\n");
            cytnx_error_msg(!Tl.is_contiguous(),"[ERROR] tensor #1 should be contiguous. suggestion: call Tensor.contiguous() or Tensor.contiguous_() first.%s","\n");
            cytnx_error_msg(!Tr.is_contiguous(),"[ERROR] tensor #2 should be contiguous. suggestion: call Tensor.contiguous() or Tensor.contiguous_() first.%s","\n");
            cytnx_error_msg(Tl.shape().size()>2,"[ERROR] tensor #1 should have either rank-1 or rank-2.%s","\n");
            cytnx_error_msg(Tr.shape().size()>2,"[ERROR] tensor #2 should have either rank-1 or rank-2.%s","\n");


            std::vector<cytnx_uint64> new_shape;
            vec_concatenate_(new_shape,Tl.shape(),Tr.shape());

            Tensor out(new_shape,Tl.dtype() < Tr.dtype()?Tl.dtype():Tr.dtype(),Tl.device());
            cytnx_uint64 i1,i2,j1,j2;
            if(Tl.shape().size()==1){i1 = 1;            j1 = Tl.shape()[0];}
            else                    {i1 = Tl.shape()[0];j1 = Tl.shape()[1];}
            if(Tr.shape().size()==1){i2 = 1;            j2 = Tr.shape()[0];}
            else                    {i2 = Tr.shape()[0];j2 = Tr.shape()[1];}


            if(Tl.device()==Device.cpu){
                cytnx::linalg_internal::lii.Outer_ii[Tl.dtype()][Tr.dtype()](out._impl->storage()._impl, Tl._impl->storage()._impl, Tr._impl->storage()._impl,i1,j1,i2,j2);
            }else{
                #ifdef UNI_GPU
                    cytnx_error_msg(true,"[Otimes] currently Otimes is not support for GPU, pending for fix.%s","\n");
                    //checkCudaErrors(cudaSetDevice(Tl.device()));
                    //cytnx::linalg_internal::lii.cuOuter_ii[Tl.dtype()][Tr.dtype()](out._impl->storage()._impl, Tl._impl->storage()._impl, Tr._impl->storage()._impl,i1,j1,i2,j2);
                #else
                    cytnx_error_msg(true,"[Otimes] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
                #endif
            }
            

            if(matrix_form){    
                cytnx_uint32 row = 1;
                cytnx_uint32 col = 1;
                if(Tl.shape().size()==1){
                    col*= Tl.shape()[0];
                }else{
                    row*= Tl.shape()[0];
                    col*= Tl.shape()[1];
                }
                if(Tr.shape().size()==1){
                    col*= Tr.shape()[0];
                    
                }else{
                    row*= Tr.shape()[0];
                    col*= Tr.shape()[1];
                }
                out.reshape_({row,col});
            }

            return out;

        }

    }//linalg
}
