#include "Type.hpp"
#include "cytnx_error.hpp"
using namespace std;

std::string cytnx::Type_class::getname(const unsigned int &type_id){


    switch (type_id){
        case Type_class::Void:
            return string("Void");
        case Type_class::ComplexDouble:
            return string("Complex Double (Complex Float64)");
        case Type_class::ComplexFloat:
            return string("Complex Float (Complex Float32)");
        case Type_class::Double:
            return string("Double (Float64)");
        case Type_class::Float:
            return string("Float32");
        case Type_class::Int64:
            return string("Int64");
        case Type_class::Uint64:
            return string("Uint64");
        case Type_class::Int32:
            return string("Int32");
        case Type_class::Uint32:
            return string("Uint32");

        default:
            cytnx_error_msg(1,"%s","[ERROR] invalid type");
    }

}

unsigned int cytnx::Type_class::c_typename_to_id(const std::string &c_name){

    if(c_name == typeid(cytnx_complex128).name()){
        return this->ComplexDouble;
    }else if(c_name == typeid(cytnx_complex64).name()){
        return this->ComplexFloat;
    }else if(c_name == typeid(cytnx_double).name()){
        return this->Double;
    }else if(c_name == typeid(cytnx_float).name()){
        return this->Float;
    }else if(c_name == typeid(cytnx_int64).name()){
        return this->Int64;
    }else if(c_name == typeid(cytnx_uint64).name()){
        return this->Uint64;
    }else if(c_name == typeid(cytnx_int32).name()){
        return this->Int32;
    }else if(c_name == typeid(cytnx_uint32).name()){
        return this->Uint32;
    }else{
        cytnx_error_msg(1,"%s","[ERROR] invalid type");
    }

}

        
namespace cytnx{
    Type_class Type;
}
