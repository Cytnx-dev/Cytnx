#include "tn_algo/MPS.hpp"
#include "random.hpp"
#include <cmath>
#include <algorithm>
using namespace std;
namespace cytnx{
    namespace tn_algo{
        std::ostream& iMPS::Print(std::ostream &os){
            os << "MPS type : " << "[iMPS]" << endl;
            os << "Size : " << this->_TNs.size() << endl;
            os << "physBD dim : " << this->phys_dim << endl;
            os << "virtBD dim : " << this->virt_dim << endl;
            os << endl;
            return os;
        }


        void iMPS::Init(const cytnx_uint64 &N, const cytnx_uint64 &phys_dim, const cytnx_uint64 &virt_dim){
            //checking:
            cytnx_error_msg(N==0,"[ERROR][iMPS] number of site N cannot be ZERO.%s","\n");

            this->phys_dim = phys_dim;
            this->virt_dim = virt_dim;            

            const cytnx_uint64& d = phys_dim; 
            const cytnx_uint64& chi = virt_dim;
            
            this->_TNs.resize(N);

            for(cytnx_int64 k=0; k<N; k++){
                this->_TNs[k] = UniTensor(cytnx::random::normal({chi, d, chi}, 0., 1.),2);
                this->_TNs[k].set_labels({2*k,2*k+1,2*k+2});
            }
                        
        }




    }

}



