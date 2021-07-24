#include "tn_algo/MPS.hpp"

using namespace std;
namespace cytnx{
    namespace tn_algo{
        std::ostream& RegularMPS::Print(std::ostream &os){
            os << "[test][RegularMPS]" << endl;
            os << "MPS type:" << "Regular" << endl;
            os << "Size: " << this->_TNs.size() << endl;
            

            return os;
        }


        void RegularMPS::Init(const cytnx_uint64 &N, const cytnx_uint64 &phys_dim, const cytnx_uint64 &virt_dim){
            this->_TNs.resize(N);
            
        }


    }

}



