#include "Type.hpp"
#include "cytnx_error.hpp"
#ifdef UNI_MKL 
    #include <mkl.h>
    namespace cytnx{
        int __blasINTsize__ = sizeof(MKL_INT);
    }
#else
    #include <lapacke.h>
    namespace cytnx{
        int __blasINTsize__ = sizeof(lapack_int);
    }
#endif
using namespace std;

/*
Type_class::Type_class(){
    this->c_typeid2_cy_typeid[type_index(typeid(void))] = this->Void;
    this->c_typeid2_cy_typeid[type_index(typeid(cytnx_complex128))] = this->ComplexDouble;
    this->c_typeid2_cy_typeid[type_index(typeid(cytnx_complex64 ))] = this->ComplexFloat ;
    this->c_typeid2_cy_typeid[type_index(typeid(cytnx_double    ))] = this->Double;
    this->c_typeid2_cy_typeid[type_index(typeid(cytnx_float     ))] = this->Float ;
    this->c_typeid2_cy_typeid[type_index(typeid(cytnx_uint64    ))] = this->Uint64;
    this->c_typeid2_cy_typeid[type_index(typeid(cytnx_int64     ))] = this->Int64 ;
    this->c_typeid2_cy_typeid[type_index(typeid(cytnx_uint32    ))] = this->Uint32;
    this->c_typeid2_cy_typeid[type_index(typeid(cytnx_int32     ))] = this->Int32 ;
    this->c_typeid2_cy_typeid[type_index(typeid(cytnx_uint16    ))] = this->Uint16;
    this->c_typeid2_cy_typeid[type_index(typeid(cytnx_int16     ))] = this->Int16 ;
    this->c_typeid2_cy_typeid[type_index(typeid(cytnx_bool      ))] = this->Bool  ;

}

int c_typeindex_to_id(const std::type_index &type_idx){
    unordered_map<std::type_index,int>::iterator it;
    it = this->c_typeid2_cy_typeid.find(type_idx);
    
    if(it==this->c_typeid2_cy_typeid.end()){
        cytnx_error_msg(true,"[ERROR] invalid type!%s","\n");
    }

    return it->second;

}
*/
bool cytnx::Type_class::is_float(const unsigned int &type_id){


    switch (type_id){
        case Type_class::Void:
            return false;
        case Type_class::ComplexDouble:
            return true;
        case Type_class::ComplexFloat:
            return true;
        case Type_class::Double:
            return true;
        case Type_class::Float:
            return true;
        case Type_class::Int64:
            return false;
        case Type_class::Uint64:
            return false;
        case Type_class::Int32:
            return false;
        case Type_class::Uint32:
            return false;
        case Type_class::Int16:
            return false;
        case Type_class::Uint16:
            return false;
        case Type_class::Bool:
            return false;
        default:
            cytnx_error_msg(1,"%s","[ERROR] invalid type");
            return false;
    }

}

bool cytnx::Type_class::is_int(const unsigned int &type_id){


    switch (type_id){
        case Type_class::Void:
            return false;
        case Type_class::ComplexDouble:
            return false;
        case Type_class::ComplexFloat:
            return false;
        case Type_class::Double:
            return false;
        case Type_class::Float:
            return false;
        case Type_class::Int64:
            return true;
        case Type_class::Uint64:
            return true;
        case Type_class::Int32:
            return true;
        case Type_class::Uint32:
            return true;
        case Type_class::Int16:
            return true;
        case Type_class::Uint16:
            return true;
        case Type_class::Bool:
            return false;
        default:
            cytnx_error_msg(1,"%s","[ERROR] invalid type");
            return false;
    }

}




bool cytnx::Type_class::is_complex(const unsigned int &type_id){


    switch (type_id){
        case Type_class::Void:
            return false;
        case Type_class::ComplexDouble:
            return true;
        case Type_class::ComplexFloat:
            return true;
        case Type_class::Double:
            return false;
        case Type_class::Float:
            return false;
        case Type_class::Int64:
            return false;
        case Type_class::Uint64:
            return true;
        case Type_class::Int32:
            return false;
        case Type_class::Uint32:
            return true;
        case Type_class::Int16:
            return false;
        case Type_class::Uint16:
            return false;
        case Type_class::Bool:
            return false;
        default:
            cytnx_error_msg(1,"%s","[ERROR] invalid type");
            return false;
    }

}

bool cytnx::Type_class::is_unsigned(const unsigned int &type_id){


    switch (type_id){
        case Type_class::Void:
            return true;
        case Type_class::ComplexDouble:
            return false;
        case Type_class::ComplexFloat:
            return false;
        case Type_class::Double:
            return false;
        case Type_class::Float:
            return false;
        case Type_class::Int64:
            return false;
        case Type_class::Uint64:
            return true;
        case Type_class::Int32:
            return false;
        case Type_class::Uint32:
            return true;
        case Type_class::Int16:
            return false;
        case Type_class::Uint16:
            return true;
        case Type_class::Bool:
            return true;
        default:
            cytnx_error_msg(1,"%s","[ERROR] invalid type");
            return false;
    }

}

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
        case Type_class::Int16:
            return string("Int16");
        case Type_class::Uint16:
            return string("Uint16");
        case Type_class::Bool:
            return string("Bool");
        default:
            cytnx_error_msg(1,"%s","[ERROR] invalid type");
            return string("");
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
    }else if(c_name == typeid(cytnx_int16).name()){
        return this->Int16;
    }else if(c_name == typeid(cytnx_uint16).name()){
        return this->Uint16;
    }else if(c_name == typeid(cytnx_bool).name()){
        return this->Bool;
    }else{
        cytnx_error_msg(1,"%s","[ERROR] invalid type");
        return 0;
    }

}

unsigned int cytnx::Type_class::typeSize(const unsigned int &type_id){


    switch (type_id){
        case Type_class::Void:
            return 0;
        case Type_class::ComplexDouble:
            return sizeof(cytnx_complex128);
        case Type_class::ComplexFloat:
            return sizeof(cytnx_complex64);
        case Type_class::Double:
            return sizeof(cytnx_double);
        case Type_class::Float:
            return sizeof(cytnx_float);
        case Type_class::Int64:
            return sizeof(cytnx_int64);
        case Type_class::Uint64:
            return sizeof(cytnx_uint64);
        case Type_class::Int32:
            return sizeof(cytnx_int32);
        case Type_class::Uint32:
            return sizeof(cytnx_uint32);
        case Type_class::Int16:
            return sizeof(cytnx_int16);
        case Type_class::Uint16:
            return sizeof(cytnx_uint16);
        case Type_class::Bool:
            return sizeof(cytnx_bool);
        default:
            cytnx_error_msg(1,"%s","[ERROR] invalid type");
            return 0;
    }

}
        
namespace cytnx{
    Type_class Type;
}
