#include "utils/is.hpp"

namespace cytnx{
    
    bool is(const Tensor &L, const Tensor &R){
        return (L._impl == R._impl);
    }

    bool is(const Storage &L, const Storage &R){
        return (L._impl == R._impl);
    }
    #ifdef EXT_Enable
    bool is(const cytnx_extension::Bond &L, const cytnx_extension::Bond &R){
        return (L._impl == R._impl);
    }

    bool is(const cytnx_extension::Symmetry &L, const cytnx_extension::Symmetry &R){
        return (L._impl == R._impl);
    }
    #endif    


}


