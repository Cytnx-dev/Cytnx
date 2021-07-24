#include "tn_algo/MPS.hpp"

namespace cytnx{
    namespace tn_algo{
        std::ostream& operator<<(std::ostream& os, const MPS& in){
            in._impl->Print(os);
            return os;
        }


    }

}



