#ifdef EXT_Enable
#include <typeinfo>
#include "extension/UniTensor.hpp"
#include "utils/utils_internal_interface.hpp"
#include "linalg.hpp"

using namespace std;

namespace cytnx{

    // += 
    template<> UniTensor& UniTensor::operator+=<UniTensor>(const UniTensor &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_complex128>(const cytnx_complex128 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_complex64>(const cytnx_complex64 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_double>(const cytnx_double &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_float>(const cytnx_float &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_int64>(const cytnx_int64 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_uint64>(const cytnx_uint64 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_int32>(const cytnx_int32 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_uint32>(const cytnx_uint32 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_int16>(const cytnx_int16 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_uint16>(const cytnx_uint16 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_bool>(const cytnx_bool &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }

    // -= 
    template<> UniTensor& UniTensor::operator-=<UniTensor>(const UniTensor &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_complex128>(const cytnx_complex128 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_complex64>(const cytnx_complex64 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_double>(const cytnx_double &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_float>(const cytnx_float &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_int64>(const cytnx_int64 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_uint64>(const cytnx_uint64 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_int32>(const cytnx_int32 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_uint32>(const cytnx_uint32 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_int16>(const cytnx_int16 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_uint16>(const cytnx_uint16 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_bool>(const cytnx_bool &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }

    // *= 
    template<> UniTensor& UniTensor::operator*=<UniTensor>(const UniTensor &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_complex128>(const cytnx_complex128 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_complex64>(const cytnx_complex64 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_double>(const cytnx_double &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_float>(const cytnx_float &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_int64>(const cytnx_int64 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_uint64>(const cytnx_uint64 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_int32>(const cytnx_int32 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_uint32>(const cytnx_uint32 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_int16>(const cytnx_int16 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_uint16>(const cytnx_uint16 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_bool>(const cytnx_bool &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }

    // /=
    template<> UniTensor& UniTensor::operator/=<UniTensor>(const UniTensor &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_complex128>(const cytnx_complex128 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_complex64>(const cytnx_complex64 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_double>(const cytnx_double &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_float>(const cytnx_float &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_int64>(const cytnx_int64 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_uint64>(const cytnx_uint64 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_int32>(const cytnx_int32 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_uint32>(const cytnx_uint32 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_int16>(const cytnx_int16 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_uint16>(const cytnx_uint16 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_bool>(const cytnx_bool &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }


}
#endif
