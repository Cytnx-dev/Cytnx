#include "utils/is_same.hpp"

namespace cytnx{
    
    template<>
    bool is_same<Tensor>(const Tensor &L, const Tensor &R){
        return (L._impl == R._impl);
    }
    template<>
    bool is_same<Storage>(const Storage &L, const Storage &R){
        return (L._impl == R._impl);
    }


}


