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

    }

}



