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

#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>

#ifdef BACKEND_TORCH
typedef int32_t blas_int;
#else

  #ifdef UNI_MKL
    #include <mkl.h>
typedef MKL_INT blas_int;
  #else
typedef int32_t blas_int;
  #endif

#endif

namespace cytnx {
  // // using namespace boost::container;

  // //This option specifies the desired alignment for the internal value_type
  // typedef boost::container::small_vector_options<boost::container::inplace_alignment<16u>>::type
  // alignment_16_option_t;
  // // //This option specifies that a vector will increase its capacity 50%
  // // //each time the previous capacity was exhausted.
  // // typedef small_vector_options< growth_factor<growth_factor_50> >::type growth_50_option_t;
  // // //Check 16 byte alignment option
  // // small_vector<int, 10, void, alignment_16_option_t > sv;
  // // //Fill the vector until full capacity is reached
  // // small_vector<int, 10, void, growth_50_option_t > growth_50_vector(10, 0);

  // template <class T, size_t N = 16>
  // using smallvec = boost::container::small_vector<T, N>;

  template <class T>
  using vec3d = std::vector<std::vector<std::vector<T>>>;

  template <class T>
  using vec2d = std::vector<std::vector<T>>;

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
    // char name[35];
    bool is_unsigned;
    bool is_complex;
    bool is_float;
    bool is_int;
    unsigned int typeSize;
  };

  static const int N_Type = 12;
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
    // std::vector<Type_struct> Typeinfos;
    inline static Type_struct Typeinfos[N_Type];
    inline static bool inited = false;
    Type_class &operator=(const Type_class &rhs) {
      for (int i = 0; i < N_Type; i++) this->Typeinfos[i] = rhs.Typeinfos[i];
      return *this;
    }

    Type_class() {
      // #ifdef DEBUG
      //   std::cout << "[DEBUG] Type constructor call. " << std::endl;
      // #endif
      if (!inited) {
        Typeinfos[this->Void] = (Type_struct){"Void", true, false, false, false, 0};
        Typeinfos[this->ComplexDouble] = (Type_struct){
          "Complex Double (Complex Float64)", false, true, true, false, sizeof(cytnx_complex128)};
        Typeinfos[this->ComplexFloat] = (Type_struct){
          "Complex Float (Complex Float32)", false, true, true, false, sizeof(cytnx_complex64)};
        Typeinfos[this->Double] =
          (Type_struct){"Double (Float64)", false, false, true, false, sizeof(cytnx_double)};
        Typeinfos[this->Float] =
          (Type_struct){"Float (Float32)", false, false, true, false, sizeof(cytnx_float)};
        Typeinfos[this->Int64] =
          (Type_struct){"Int64", false, false, false, true, sizeof(cytnx_int64)};
        Typeinfos[this->Uint64] =
          (Type_struct){"Uint64", true, false, false, true, sizeof(cytnx_uint64)};
        Typeinfos[this->Int32] =
          (Type_struct){"Int32", false, false, false, true, sizeof(cytnx_int32)};
        Typeinfos[this->Uint32] =
          (Type_struct){"Uint32", true, false, false, true, sizeof(cytnx_uint32)};
        Typeinfos[this->Int16] =
          (Type_struct){"Int16", false, false, false, true, sizeof(cytnx_int16)};
        Typeinfos[this->Uint16] =
          (Type_struct){"Uint16", true, false, false, true, sizeof(cytnx_uint16)};
        Typeinfos[this->Bool] =
          (Type_struct){"Bool", true, false, false, false, sizeof(cytnx_bool)};

        inited = true;
      }
    }
    const std::string &getname(const unsigned int &type_id) const;
    unsigned int c_typename_to_id(const std::string &c_name) const;
    unsigned int typeSize(const unsigned int &type_id) const;
    bool is_unsigned(const unsigned int &type_id) const;
    bool is_complex(const unsigned int &type_id) const;
    bool is_float(const unsigned int &type_id) const;
    bool is_int(const unsigned int &type_id) const;
    // int c_typeindex_to_id(const std::type_index &type_idx);
    template <class T>
    unsigned int cy_typeid(const T &rc) const {
      cytnx_error_msg(true, "[ERROR] invalid type%s", "\n");
      return 0;
    }
    static unsigned int cy_typeid(const cytnx_complex128 &rc) { return Type_class::ComplexDouble; }
    static unsigned int cy_typeid(const cytnx_complex64 &rc) { return Type_class::ComplexFloat; }
    static unsigned int cy_typeid(const cytnx_double &rc) { return Type_class::Double; }
    static unsigned int cy_typeid(const cytnx_float &rc) { return Type_class::Float; }
    static unsigned int cy_typeid(const cytnx_uint64 &rc) { return Type_class::Uint64; }
    static unsigned int cy_typeid(const cytnx_int64 &rc) { return Type_class::Int64; }
    static unsigned int cy_typeid(const cytnx_uint32 &rc) { return Type_class::Uint32; }
    static unsigned int cy_typeid(const cytnx_int32 &rc) { return Type_class::Int32; }
    static unsigned int cy_typeid(const cytnx_uint16 &rc) { return Type_class::Uint16; }
    static unsigned int cy_typeid(const cytnx_int16 &rc) { return Type_class::Int16; }
    static unsigned int cy_typeid(const cytnx_bool &rc) { return Type_class::Bool; }

    unsigned int type_promote(const unsigned int &typeL, const unsigned int &typeR);
  };
  /// @endcond

  /// @cond
  int type_promote(const int &typeL, const int &typeR);
  /// @endcond

  /**
   * @brief data type
   *
   * @details This is the variable about the data type of the UniTensor, Tensor, ... .\n
   *     You can use it as following:
   *     \code
   *     int type = Type.Double;
   *     \endcode
   *
   *     The supported enumerations are as following:
   *
   *  enumeration  |  description
   * --------------|--------------------
   *  Void         |  the data type is void (nothing)
   *  ComplexDouble|  complex double type with 128 bits
   *  ComplexFloat |  complex float type with 64 bits
   *  Double       |  double float type with 64 bits
   *  Float        |  single float type with 32 bits
   *  Int64        |  long long integer type with 64 bits
   *  Uint64       |  unsigned long long integer type with 64 bits
   *  Int32        |  integer type with 32 bits
   *  Uint32       |  unsigned integer type with 32 bits
   *  Int16        |  short integer type with 16 bits
   *  Uint16       |  undigned short integer with 16 bits
   *  Bool         |  boolean type
   */
  extern Type_class Type;  // move to cytnx.hpp and guarded
  // static const Type_class Type = Type_class();

  extern int __blasINTsize__;

  extern bool User_debug;

}  // namespace cytnx

#endif
