#include "Scalar.hpp"

namespace cytnx{

    cytnx_complex128 complex128(const Scalar &in){
        return in._impl->to_cytnx_complex128();
    }

    cytnx_complex64 complex64(const Scalar &in){
        return in._impl->to_cytnx_complex64();
    }

    std::ostream& operator<<(std::ostream& os, const Scalar &in){
        os << std::string("Scalar dtype: [") << Type.getname(in._impl->_dtype) << std::string("]") << std::endl;
        in._impl->print(os);
        return os;
    }

    


}


