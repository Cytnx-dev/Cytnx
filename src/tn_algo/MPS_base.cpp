#include "tn_algo/MPS.hpp"

using namespace std;
namespace cytnx{
    namespace tn_algo{
        std::ostream& MPS_impl::Print(std::ostream &os){
            cytnx_error_msg(true,"[ERROR] MPS_Base should not be called. Please initialize the MPS first.%s","\n");
            return os;
        }

    }

}



