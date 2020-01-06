#include "utils/is.hpp"

namespace cytnx{
    
    bool is(const Tensor &L, const Tensor &R){
        return (L._impl == R._impl);
    }

    bool is(const Storage &L, const Storage &R){
        return (L._impl == R._impl);
    }
    


}


