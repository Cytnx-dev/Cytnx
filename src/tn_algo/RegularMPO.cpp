#include "tn_algo/MPO.hpp"

using namespace std;
namespace cytnx{
    namespace tn_algo{
        std::ostream& RegularMPO::Print(std::ostream &os){
            os << "[test][RegularMPO]" << endl;
            os << "MPO type:" << "Regular" << endl;
            os << "Number of Op: " << this->_TNs.size() << endl;
            

            return os;
        }

    }

}



