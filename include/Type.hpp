#ifndef _H_TYPE_
#define _H_TYPE_

#include <string>
#include <complex>
#include <stdint.h>
#include <climits>




namespace tor10{
    typedef double tor10_double;
    typedef float tor10_float;
    typedef uint64_t tor10_uint64;
    typedef uint32_t tor10_uint32;
    typedef int64_t tor10_int64;
    typedef int32_t tor10_int32;
    typedef std::complex<float> tor10_complex64;
    typedef std::complex<double> tor10_complex128;


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
    class Type{
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
             

    };

    extern Type tor10type;
}//namespace tor10

#endif
