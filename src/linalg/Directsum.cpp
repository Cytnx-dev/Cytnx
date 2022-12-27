#include "linalg.hpp"
#include "utils/utils.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
#include "cytnx.hpp"
#ifdef UNI_OMP
  #include <omp.h>
#endif

using namespace std;
namespace cytnx {
  namespace linalg {

    cytnx::Tensor Directsum(const cytnx::Tensor &T1, const cytnx::Tensor &T2, const std::vector<cytnx_uint64> &shared_axes) {
        
        //check:
        cytnx_error_msg(T1.shape().size()!=T2.shape().size(),"[ERROR] T1 and T2 must be the same rank!%s","\n");
        cytnx_error_msg(shared_axes.size()>T1.shape().size(),"[ERROR] len(shared_axes) must be small or equal to the rank of Tensors.%s","\n");
        cytnx_error_msg(T1.device()!=T2.device(),"[ERROR] Two tensors must be on same devices.%s","\n");
        
        //checking dulipcation in shared_axes:
        auto tmp = vec_unique(shared_axes); 
        cytnx_error_msg(tmp.size()!=shared_axes.size(),"[ERROR] shared_axes cannot contain duplicate elements!%s","\n");

        // new shape holder
        std::vector<cytnx_uint64> new_shape(T1.rank());

        //checking dimension in shared_axes:
        for(int i=0;i<shared_axes.size();i++){
            cytnx_error_msg(shared_axes[i] >= T1.shape().size(),"[ERROR] axis %d specify in shared_axes[%d] is out of bound!\n",shared_axes[i],i);
            cytnx_error_msg(T1.shape()[shared_axes[i]]!= T2.shape()[shared_axes[i]],"[ERROR] T1 and T2 at axis %d which specified to share does not have same dimension!\n",shared_axes[i]);
            
            new_shape[shared_axes[i]] = T1.shape()[shared_axes[i]];
        }


      Tensor _t1 = T1.contiguous(), _t2 = T2.contiguous();
      if (T1.dtype() != T2.dtype()) {
        // do conversion:
        if (T1.dtype() < T2.dtype()) {
          _t2 = _t2.astype(T1.dtype());
        } else {
          _t1 = _t1.astype(T2.dtype());
        }
      }

      //calculate new shape:
      for(int i=0;i<new_shape.size();i++){
        if(new_shape[i]==0){
            new_shape[i] = T1.shape()[i] + T2.shape()[i];
        }
      }

      
      Tensor out(new_shape,_t1.dtype(),_t1.device());

        


      return out; 


    }
  


  }  // namespace linalg
}  // namespace cytnx

