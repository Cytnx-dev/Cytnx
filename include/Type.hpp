#ifndef _H_TYPE_
#define _H_TYPE_

#include <string>
#include <complex>
#include <vector>
#include <stdint.h>
#include <climits>
#include <typeinfo>
#include <unordered_map>
#include <typeindex>
#include "cytnx_error.hpp"
//#ifdef UNI_MKL
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>

//#endif

namespace cytnx {
  typedef double cytnx_double;
  typedef float cytnx_float;
  typedef uint64_t cytnx_uint64;
  typedef uint32_t cytnx_uint32;
  typedef uint16_t cytnx_uint16;
  typedef int64_t cytnx_int64;
  typedef int32_t cytnx_int32;
  typedef int16_t cytnx_int16;
  typedef size_t cytnx_size_t;
  typedef std::complex<float> cytnx_complex64;
  typedef std::complex<double> cytnx_complex128;
  typedef bool cytnx_bool;

  /// @cond
  struct __type {
    enum __pybind_type {
      Void,
      ComplexDouble,
      ComplexFloat,
      Double,
      Float,
      Int64,
      Uint64,
      Int32,
      Uint32,
      Int16,
      Uint16,
      Bool,
    };
  };

  struct Type_struct {
    std::string name;
    bool is_unsigned;
    bool is_complex;
    bool is_float;
    bool is_int;
    unsigned int typeSize;
  };

  const int N_Type = 12;
  const int N_fType = 5;

  class Type_class {
   private:
   public:
    enum : unsigned int {
      Void,
      ComplexDouble,
      ComplexFloat,
      Double,
      Float,
      Int64,
      Uint64,
      Int32,
      Uint32,
      Int16,
      Uint16,
      Bool
    };
    std::vector<Type_struct> Typeinfos;

    Type_class();
    const std::string &getname(const unsigned int &type_id);
    unsigned int c_typename_to_id(const std::string &c_name);
    unsigned int typeSize(const unsigned int &type_id);
    bool is_unsigned(const unsigned int &type_id);
    bool is_complex(const unsigned int &type_id);
    bool is_float(const unsigned int &type_id);
    bool is_int(const unsigned int &type_id);
    // int c_typeindex_to_id(const std::type_index &type_idx);
    template <class T>
    unsigned int cy_typeid(const T &rc) {
      cytnx_error_msg(true, "[ERROR] invalid type%s", "\n");
      return 0;
    }
    unsigned int cy_typeid(const cytnx_complex128 &rc) { return Type_class::ComplexDouble; }
    unsigned int cy_typeid(const cytnx_complex64 &rc) { return Type_class::ComplexFloat; }
    unsigned int cy_typeid(const cytnx_double &rc) { return Type_class::Double; }
    unsigned int cy_typeid(const cytnx_float &rc) { return Type_class::Float; }
    unsigned int cy_typeid(const cytnx_uint64 &rc) { return Type_class::Uint64; }
    unsigned int cy_typeid(const cytnx_int64 &rc) { return Type_class::Int64; }
    unsigned int cy_typeid(const cytnx_uint32 &rc) { return Type_class::Uint32; }
    unsigned int cy_typeid(const cytnx_int32 &rc) { return Type_class::Int32; }
    unsigned int cy_typeid(const cytnx_uint16 &rc) { return Type_class::Uint16; }
    unsigned int cy_typeid(const cytnx_int16 &rc) { return Type_class::Int16; }
    unsigned int cy_typeid(const cytnx_bool &rc) { return Type_class::Bool; }
  };
  /// @endcond

  extern Type_class Type;
  extern int __blasINTsize__;

  extern bool User_debug;

}  // namespace cytnx

#endif
