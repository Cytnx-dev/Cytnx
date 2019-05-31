#ifndef _H_TYPE_
#define _H_TYPE_

#include <string>
#include <complex>
#include <stdint.h>
#include <climits>




namespace cytnx{
    typedef double cytnx_double;
    typedef float cytnx_float;
    typedef uint64_t cytnx_uint64;
    typedef uint32_t cytnx_uint32;
    typedef int64_t cytnx_int64;
    typedef int32_t cytnx_int32;
    typedef std::complex<float> cytnx_complex64;
    typedef std::complex<double> cytnx_complex128;


    struct __type{
        enum __pybind_type{
            Void,
            ComplexDouble,
            ComplexFloat,
            Double,
            Float,
            Int64,
            Uint64,
            Int32,
            Uint32
        };
    };


    const int N_Type=9;
    class Type_class{
        public:
            enum:unsigned int{
                Void,
                ComplexDouble,
                ComplexFloat,
                Double,
                Float,
                Int64,
                Uint64,
                Int32,
                Uint32
            };

            
            std::string getname(const unsigned int &type_id);
            unsigned int c_typename_to_id(const std::string &c_name);

    };

    extern Type_class Type;

}//namespace cytnx






#endif
