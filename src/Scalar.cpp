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

    //Storage Init interface.    
    //=============================
    Scalar_base* ScIInit_cd(){
        Scalar_base* out = new ComplexDoubleScalar();
        return out;
    }
    Scalar_base* ScIInit_cf(){
        Scalar_base* out = new ComplexFloatScalar();
        return out;
    }
    Scalar_base* ScIInit_d(){
        Scalar_base* out = new DoubleScalar();
        return out;
    }
    Scalar_base* ScIInit_f(){
        Scalar_base* out = new FloatScalar();
        return out;
    }
    Scalar_base* ScIInit_u64(){
        Scalar_base* out = new Uint64Scalar();
        return out;
    }
    Scalar_base* ScIInit_i64(){
        Scalar_base* out = new Int64Scalar();
        return out;
    }
    Scalar_base* ScIInit_u32(){
        Scalar_base* out = new Uint32Scalar();
        return out;
    }
    Scalar_base* ScIInit_i32(){
        Scalar_base* out = new Int32Scalar();
        return out;
    }
    Scalar_base* ScIInit_u16(){
        Scalar_base* out = new Uint16Scalar();
        return out;
    }
    Scalar_base* ScIInit_i16(){
        Scalar_base* out = new Int16Scalar();
        return out;
    }
    Scalar_base* ScIInit_b(){
        Scalar_base* out = new BoolScalar();
        return out;
    }
    Scalar_init_interface::Scalar_init_interface(){
        UScIInit.resize(N_Type);
        UScIInit[this->Double] = ScIInit_d;
        UScIInit[this->Float] = ScIInit_f;
        UScIInit[this->ComplexDouble] = ScIInit_cd;
        UScIInit[this->ComplexFloat] = ScIInit_cf;
        UScIInit[this->Uint64] = ScIInit_u64;
        UScIInit[this->Int64] = ScIInit_i64;
        UScIInit[this->Uint32]= ScIInit_u32;
        UScIInit[this->Int32] = ScIInit_i32;
        UScIInit[this->Uint16]= ScIInit_u16;
        UScIInit[this->Int16] = ScIInit_i16;
        UScIInit[this->Bool] = ScIInit_b;
    }


    Scalar_init_interface __ScII;



}


