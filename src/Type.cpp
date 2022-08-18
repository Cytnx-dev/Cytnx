#include "Type.hpp"
#include "cytnx_error.hpp"
#ifdef UNI_MKL
  #include <mkl.h>
namespace cytnx {
  int __blasINTsize__ = sizeof(MKL_INT);
}
#else
  #include <lapacke.h>
namespace cytnx {
  int __blasINTsize__ = sizeof(lapack_int);
}
#endif

// global debug flag!
namespace cytnx {
  bool User_debug = true;
}

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

cytnx::Type_class::Type_class() {
  Typeinfos.resize(N_Type);
  //{name,unsigned,complex,float,int,typesize}
  Typeinfos[this->Void] = (Type_struct){std::string("Void"), true, false, false, false, 0};
  Typeinfos[this->ComplexDouble] = (Type_struct){std::string("Complex Double (Complex Float64)"),
                                                 false,
                                                 true,
                                                 true,
                                                 false,
                                                 sizeof(cytnx_complex128)};
  Typeinfos[this->ComplexFloat] = (Type_struct){std::string("Complex Float (Complex Float32)"),
                                                false,
                                                true,
                                                true,
                                                false,
                                                sizeof(cytnx_complex64)};
  Typeinfos[this->Double] =
    (Type_struct){std::string("Double (Float64)"), false, false, true, false, sizeof(cytnx_double)};
  Typeinfos[this->Float] =
    (Type_struct){std::string("Float (Float32)"), false, false, true, false, sizeof(cytnx_float)};
  Typeinfos[this->Int64] =
    (Type_struct){std::string("Int64"), false, false, false, true, sizeof(cytnx_int64)};
  Typeinfos[this->Uint64] =
    (Type_struct){std::string("Uint64"), true, false, false, true, sizeof(cytnx_uint64)};
  Typeinfos[this->Int32] =
    (Type_struct){std::string("Int32"), false, false, false, true, sizeof(cytnx_int32)};
  Typeinfos[this->Uint32] =
    (Type_struct){std::string("Uint32"), true, false, false, true, sizeof(cytnx_uint32)};
  Typeinfos[this->Int16] =
    (Type_struct){std::string("Int16"), false, false, false, true, sizeof(cytnx_int16)};
  Typeinfos[this->Uint16] =
    (Type_struct){std::string("Uint16"), true, false, false, true, sizeof(cytnx_uint16)};
  Typeinfos[this->Bool] =
    (Type_struct){std::string("Bool"), true, false, false, false, sizeof(cytnx_bool)};
}

bool cytnx::Type_class::is_float(const unsigned int &type_id) {
  cytnx_error_msg(type_id >= N_Type, "[ERROR] invalid type_id%s", "\n");
  return Typeinfos[type_id].is_float;
}

bool cytnx::Type_class::is_int(const unsigned int &type_id) {
  cytnx_error_msg(type_id >= N_Type, "[ERROR] invalid type_id%s", "\n");
  return Typeinfos[type_id].is_int;
}

bool cytnx::Type_class::is_complex(const unsigned int &type_id) {
  cytnx_error_msg(type_id >= N_Type, "[ERROR] invalid type_id%s", "\n");
  return Typeinfos[type_id].is_complex;
}

bool cytnx::Type_class::is_unsigned(const unsigned int &type_id) {
  cytnx_error_msg(type_id >= N_Type, "[ERROR] invalid type_id%s", "\n");
  return Typeinfos[type_id].is_unsigned;
}

const std::string &cytnx::Type_class::getname(const unsigned int &type_id) {
  cytnx_error_msg(type_id >= N_Type, "[ERROR] invalid type_id%s", "\n");
  return Typeinfos[type_id].name;
}
unsigned int cytnx::Type_class::typeSize(const unsigned int &type_id) {
  cytnx_error_msg(type_id >= N_Type, "[ERROR] invalid type_id%s", "\n");
  return Typeinfos[type_id].typeSize;
}
unsigned int cytnx::Type_class::c_typename_to_id(const std::string &c_name) {
  if (c_name == typeid(cytnx_complex128).name()) {
    return Type_class::ComplexDouble;
  } else if (c_name == typeid(cytnx_complex64).name()) {
    return Type_class::ComplexFloat;
  } else if (c_name == typeid(cytnx_double).name()) {
    return Type_class::Double;
  } else if (c_name == typeid(cytnx_float).name()) {
    return Type_class::Float;
  } else if (c_name == typeid(cytnx_int64).name()) {
    return Type_class::Int64;
  } else if (c_name == typeid(cytnx_uint64).name()) {
    return Type_class::Uint64;
  } else if (c_name == typeid(cytnx_int32).name()) {
    return Type_class::Int32;
  } else if (c_name == typeid(cytnx_uint32).name()) {
    return Type_class::Uint32;
  } else if (c_name == typeid(cytnx_int16).name()) {
    return Type_class::Int16;
  } else if (c_name == typeid(cytnx_uint16).name()) {
    return Type_class::Uint16;
  } else if (c_name == typeid(cytnx_bool).name()) {
    return Type_class::Bool;
  } else {
    cytnx_error_msg(1, "%s", "[ERROR] invalid type");
    return 0;
  }
}

namespace cytnx {
  Type_class Type;
}
