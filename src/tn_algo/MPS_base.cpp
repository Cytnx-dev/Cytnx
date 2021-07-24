#include "tn_algo/MPS.hpp"

using namespace std;
namespace cytnx{
    namespace tn_algo{
        std::ostream& MPS_impl::Print(std::ostream &os){
            cytnx_error_msg(true,"[ERROR] MPS_Base should not be called. Please initialize the MPS first.%s","\n");
            return os;
        }

        void MPS_impl::Init(const cytnx_uint64 &N , const cytnx_uint64 &phys_dim, const cytnx_uint64 &virt_dim){
            cytnx_error_msg(true,"[ERROR] MPS_Base should not be called. Please initialize the MPS first.%s","\n");
        }

    }

}



