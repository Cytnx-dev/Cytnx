#ifndef __complex_arithmic__H_
#define __complex_arithmic__H_

#include "Type.hpp"
#include "tor10_error.hpp"


namespace tor10{

// operator overload for CPU code. for COMPLEX type arithmic.
    //tor10_complex128 operator+(const tor10_complex128 &ln, const tor10_complex128 &rn);
    tor10_complex128 operator+(const tor10_complex128 &ln, const tor10_complex64 &rn);
    tor10_complex128 operator+(const tor10_complex128 &ln, const tor10_double &rn);
    tor10_complex128 operator+(const tor10_complex128 &ln, const tor10_float &rn);
    tor10_complex128 operator+(const tor10_complex128 &ln, const tor10_uint64 &rn);
    tor10_complex128 operator+(const tor10_complex128 &ln, const tor10_uint32 &rn);
    tor10_complex128 operator+(const tor10_complex128 &ln, const tor10_int64 &rn);
    tor10_complex128 operator+(const tor10_complex128 &ln, const tor10_int32 &rn);

    tor10_complex128 operator+(const tor10_complex64 &ln, const tor10_complex128 &rn);
    //tor10_complex64 operator+(const tor10_complex64 &ln, const tor10_complex64 &rn);
    tor10_complex64 operator+(const tor10_complex64 &ln, const tor10_double &rn);
    tor10_complex64 operator+(const tor10_complex64 &ln, const tor10_float &rn);
    tor10_complex64 operator+(const tor10_complex64 &ln, const tor10_uint64 &rn);
    tor10_complex64 operator+(const tor10_complex64 &ln, const tor10_uint32 &rn);
    tor10_complex64 operator+(const tor10_complex64 &ln, const tor10_int64 &rn);
    tor10_complex64 operator+(const tor10_complex64 &ln, const tor10_int32 &rn);

    //tor10_complex128 operator+(const tor10_complex128 &rn,const tor10_complex128 &ln);
    //tor10_complex128 operator+(const tor10_complex64 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator+(const tor10_double &rn,const tor10_complex128 &ln);
    tor10_complex128 operator+(const tor10_float &rn,const tor10_complex128 &ln);
    tor10_complex128 operator+(const tor10_uint64 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator+(const tor10_uint32 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator+(const tor10_int64 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator+(const tor10_int32 &rn,const tor10_complex128 &ln);

    //tor10_complex128 operator+(const tor10_complex128 &rn,const tor10_complex64 &ln);
    //tor10_complex64 operator+(const tor10_complex64 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator+(const tor10_double &rn,const tor10_complex64 &ln);
    tor10_complex64 operator+( const tor10_float &rn,const tor10_complex64 &ln);
    tor10_complex64 operator+(const tor10_uint64 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator+(const tor10_uint32 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator+(const tor10_int64 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator+( const tor10_int32 &rn,const tor10_complex64 &ln);

//---------------------
    //tor10_complex128 operator-(const tor10_complex128 &ln, const tor10_complex128 &rn);
    tor10_complex128 operator-(const tor10_complex128 &ln, const tor10_complex64 &rn);
    tor10_complex128 operator-(const tor10_complex128 &ln, const tor10_double &rn);
    tor10_complex128 operator-(const tor10_complex128 &ln, const tor10_float &rn);
    tor10_complex128 operator-(const tor10_complex128 &ln, const tor10_uint64 &rn);
    tor10_complex128 operator-(const tor10_complex128 &ln, const tor10_uint32 &rn);
    tor10_complex128 operator-(const tor10_complex128 &ln, const tor10_int64 &rn);
    tor10_complex128 operator-(const tor10_complex128 &ln, const tor10_int32 &rn);

    tor10_complex128 operator-(const tor10_complex64 &ln, const tor10_complex128 &rn);
    //tor10_complex64 operator-(const tor10_complex64 &ln, const tor10_complex64 &rn);
    tor10_complex64 operator-(const tor10_complex64 &ln, const tor10_double &rn);
    tor10_complex64 operator-(const tor10_complex64 &ln, const tor10_float &rn);
    tor10_complex64 operator-(const tor10_complex64 &ln, const tor10_uint64 &rn);
    tor10_complex64 operator-(const tor10_complex64 &ln, const tor10_uint32 &rn);
    tor10_complex64 operator-(const tor10_complex64 &ln, const tor10_int64 &rn);
    tor10_complex64 operator-(const tor10_complex64 &ln, const tor10_int32 &rn);

    //tor10_complex128 operator-(const tor10_complex128 &rn,const tor10_complex128 &ln);
    //tor10_complex128 operator-(const tor10_complex64 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator-(const tor10_double &rn,const tor10_complex128 &ln);
    tor10_complex128 operator-(const tor10_float &rn,const tor10_complex128 &ln);
    tor10_complex128 operator-(const tor10_uint64 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator-(const tor10_uint32 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator-(const tor10_int64 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator-(const tor10_int32 &rn,const tor10_complex128 &ln);

    //tor10_complex128 operator-(const tor10_complex128 &rn,const tor10_complex64 &ln);
    //tor10_complex64 operator-(const tor10_complex64 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator-(const tor10_double &rn,const tor10_complex64 &ln);
    tor10_complex64 operator-( const tor10_float &rn,const tor10_complex64 &ln);
    tor10_complex64 operator-(const tor10_uint64 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator-(const tor10_uint32 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator-(const tor10_int64 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator-( const tor10_int32 &rn,const tor10_complex64 &ln);

//---------------------
    //tor10_complex128 operator*(const tor10_complex128 &ln, const tor10_complex128 &rn);
    tor10_complex128 operator*(const tor10_complex128 &ln, const tor10_complex64 &rn);
    tor10_complex128 operator*(const tor10_complex128 &ln, const tor10_double &rn);
    tor10_complex128 operator*(const tor10_complex128 &ln, const tor10_float &rn);
    tor10_complex128 operator*(const tor10_complex128 &ln, const tor10_uint64 &rn);
    tor10_complex128 operator*(const tor10_complex128 &ln, const tor10_uint32 &rn);
    tor10_complex128 operator*(const tor10_complex128 &ln, const tor10_int64 &rn);
    tor10_complex128 operator*(const tor10_complex128 &ln, const tor10_int32 &rn);

    tor10_complex128 operator*(const tor10_complex64 &ln, const tor10_complex128 &rn);
    //tor10_complex64 operator*(const tor10_complex64 &ln, const tor10_complex64 &rn);
    tor10_complex64 operator*(const tor10_complex64 &ln, const tor10_double &rn);
    tor10_complex64 operator*(const tor10_complex64 &ln, const tor10_float &rn);
    tor10_complex64 operator*(const tor10_complex64 &ln, const tor10_uint64 &rn);
    tor10_complex64 operator*(const tor10_complex64 &ln, const tor10_uint32 &rn);
    tor10_complex64 operator*(const tor10_complex64 &ln, const tor10_int64 &rn);
    tor10_complex64 operator*(const tor10_complex64 &ln, const tor10_int32 &rn);

    //tor10_complex128 operator*(const tor10_complex128 &rn,const tor10_complex128 &ln);
    //tor10_complex128 operator*(const tor10_complex64 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator*(const tor10_double &rn,const tor10_complex128 &ln);
    tor10_complex128 operator*(const tor10_float &rn,const tor10_complex128 &ln);
    tor10_complex128 operator*(const tor10_uint64 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator*(const tor10_uint32 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator*(const tor10_int64 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator*(const tor10_int32 &rn,const tor10_complex128 &ln);

    //tor10_complex128 operator*(const tor10_complex128 &rn,const tor10_complex64 &ln);
    //tor10_complex64 operator*(const tor10_complex64 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator*(const tor10_double &rn,const tor10_complex64 &ln);
    tor10_complex64 operator*( const tor10_float &rn,const tor10_complex64 &ln);
    tor10_complex64 operator*(const tor10_uint64 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator*(const tor10_uint32 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator*(const tor10_int64 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator*( const tor10_int32 &rn,const tor10_complex64 &ln);

//---------------------
    //tor10_complex128 operator/(const tor10_complex128 &ln, const tor10_complex128 &rn);
    tor10_complex128 operator/(const tor10_complex128 &ln, const tor10_complex64 &rn);
    tor10_complex128 operator/(const tor10_complex128 &ln, const tor10_double &rn);
    tor10_complex128 operator/(const tor10_complex128 &ln, const tor10_float &rn);
    tor10_complex128 operator/(const tor10_complex128 &ln, const tor10_uint64 &rn);
    tor10_complex128 operator/(const tor10_complex128 &ln, const tor10_uint32 &rn);
    tor10_complex128 operator/(const tor10_complex128 &ln, const tor10_int64 &rn);
    tor10_complex128 operator/(const tor10_complex128 &ln, const tor10_int32 &rn);

    tor10_complex128 operator/(const tor10_complex64 &ln, const tor10_complex128 &rn);
    //tor10_complex64 operator/(const tor10_complex64 &ln, const tor10_complex64 &rn);
    tor10_complex64 operator/(const tor10_complex64 &ln, const tor10_double &rn);
    tor10_complex64 operator/(const tor10_complex64 &ln, const tor10_float &rn);
    tor10_complex64 operator/(const tor10_complex64 &ln, const tor10_uint64 &rn);
    tor10_complex64 operator/(const tor10_complex64 &ln, const tor10_uint32 &rn);
    tor10_complex64 operator/(const tor10_complex64 &ln, const tor10_int64 &rn);
    tor10_complex64 operator/(const tor10_complex64 &ln, const tor10_int32 &rn);

    //tor10_complex128 operator/(const tor10_complex128 &rn,const tor10_complex128 &ln);
    //tor10_complex128 operator/(const tor10_complex64 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator/(const tor10_double &rn,const tor10_complex128 &ln);
    tor10_complex128 operator/(const tor10_float &rn,const tor10_complex128 &ln);
    tor10_complex128 operator/(const tor10_uint64 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator/(const tor10_uint32 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator/(const tor10_int64 &rn,const tor10_complex128 &ln);
    tor10_complex128 operator/(const tor10_int32 &rn,const tor10_complex128 &ln);

    //tor10_complex128 operator/(const tor10_complex128 &rn,const tor10_complex64 &ln);
    //tor10_complex64 operator/(const tor10_complex64 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator/(const tor10_double &rn,const tor10_complex64 &ln);
    tor10_complex64 operator/( const tor10_float &rn,const tor10_complex64 &ln);
    tor10_complex64 operator/(const tor10_uint64 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator/(const tor10_uint32 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator/(const tor10_int64 &rn,const tor10_complex64 &ln);
    tor10_complex64 operator/( const tor10_int32 &rn,const tor10_complex64 &ln);


// operator overload for GPU code. for COMPLEX type arithmic.
} // namespace tor10


// device complex op. 
/*
namespace tor10{

#ifdef UNI_GPU
    __device__ cuDoubleComplex operator+(const cuDoubleComplex &ln, const cuDoubleComplex &rn);
    __device__ cuDoubleComplex operator+(const cuDoubleComplex &ln, const cuFloatComplex &rn);
    __device__ cuDoubleComplex operator+(const cuDoubleComplex &ln, const tor10_double &rn);
    __device__ cuDoubleComplex operator+(const cuDoubleComplex &ln, const tor10_float &rn);
    __device__ cuDoubleComplex operator+(const cuDoubleComplex &ln, const tor10_uint64 &rn);
    __device__ cuDoubleComplex operator+(const cuDoubleComplex &ln, const tor10_uint32 &rn);
    __device__ cuDoubleComplex operator+(const cuDoubleComplex &ln, const tor10_int64 &rn);
    __device__ cuDoubleComplex operator+(const cuDoubleComplex &ln, const tor10_int32 &rn);

    __device__ cuDoubleComplex operator+(const cuFloatComplex &ln, const cuDoubleComplex &rn);
    __device__ cuFloatComplex operator+(const cuFloatComplex &ln, const cuFloatComplex &rn);
    __device__ cuFloatComplex operator+(const cuFloatComplex &ln, const tor10_double &rn);
    __device__ cuFloatComplex operator+(const cuFloatComplex &ln, const tor10_float &rn);
    __device__ cuFloatComplex operator+(const cuFloatComplex &ln, const tor10_uint64 &rn);
    __device__ cuFloatComplex operator+(const cuFloatComplex &ln, const tor10_uint32 &rn);
    __device__ cuFloatComplex operator+(const cuFloatComplex &ln, const tor10_int64 &rn);
    __device__ cuFloatComplex operator+(const cuFloatComplex &ln, const tor10_int32 &rn);

    //__device__ cuDoubleComplex operator+(const cuDoubleComplex &rn,const cuDoubleComplex &ln);
    //__device__ cuDoubleComplex operator+(const cuFloatComplex &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator+(const tor10_double &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator+(const tor10_float &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator+(const tor10_uint64 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator+(const tor10_uint32 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator+(const tor10_int64 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator+(const tor10_int32 &rn,const cuDoubleComplex &ln);

    //__device__ cuDoubleComplex operator+(const cuDoubleComplex &rn,const cuFloatComplex &ln);
    //__device__ cuFloatComplex operator+(const cuFloatComplex &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator+(const tor10_double &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator+( const tor10_float &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator+(const tor10_uint64 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator+(const tor10_uint32 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator+(const tor10_int64 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator+( const tor10_int32 &rn,const cuFloatComplex &ln);

//----------------
    __device__ cuDoubleComplex operator-(const cuDoubleComplex &ln, const cuDoubleComplex &rn);
    __device__ cuDoubleComplex operator-(const cuDoubleComplex &ln, const cuFloatComplex &rn);
    __device__ cuDoubleComplex operator-(const cuDoubleComplex &ln, const tor10_double &rn);
    __device__ cuDoubleComplex operator-(const cuDoubleComplex &ln, const tor10_float &rn);
    __device__ cuDoubleComplex operator-(const cuDoubleComplex &ln, const tor10_uint64 &rn);
    __device__ cuDoubleComplex operator-(const cuDoubleComplex &ln, const tor10_uint32 &rn);
    __device__ cuDoubleComplex operator-(const cuDoubleComplex &ln, const tor10_int64 &rn);
    __device__ cuDoubleComplex operator-(const cuDoubleComplex &ln, const tor10_int32 &rn);

    __device__ cuDoubleComplex operator-(const cuFloatComplex &ln, const cuDoubleComplex &rn);
    __device__ cuFloatComplex operator-(const cuFloatComplex &ln, const cuFloatComplex &rn);
    __device__ cuFloatComplex operator-(const cuFloatComplex &ln, const tor10_double &rn);
    __device__ cuFloatComplex operator-(const cuFloatComplex &ln, const tor10_float &rn);
    __device__ cuFloatComplex operator-(const cuFloatComplex &ln, const tor10_uint64 &rn);
    __device__ cuFloatComplex operator-(const cuFloatComplex &ln, const tor10_uint32 &rn);
    __device__ cuFloatComplex operator-(const cuFloatComplex &ln, const tor10_int64 &rn);
    __device__ cuFloatComplex operator-(const cuFloatComplex &ln, const tor10_int32 &rn);

    //__device__ cuDoubleComplex operator-(const cuDoubleComplex &rn,const cuDoubleComplex &ln);
    //__device__ cuDoubleComplex operator-(const cuFloatComplex &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator-(const tor10_double &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator-(const tor10_float &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator-(const tor10_uint64 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator-(const tor10_uint32 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator-(const tor10_int64 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator-(const tor10_int32 &rn,const cuDoubleComplex &ln);

    //__device__ cuDoubleComplex operator-(const cuDoubleComplex &rn,const cuFloatComplex &ln);
    //__device__ cuFloatComplex operator-(const cuFloatComplex &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator-(const tor10_double &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator-( const tor10_float &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator-(const tor10_uint64 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator-(const tor10_uint32 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator-(const tor10_int64 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator-( const tor10_int32 &rn,const cuFloatComplex &ln);

//----------------
    __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cuDoubleComplex &rn);
    __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cuFloatComplex &rn);
    __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const tor10_double &rn);
    __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const tor10_float &rn);
    __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const tor10_uint64 &rn);
    __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const tor10_uint32 &rn);
    __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const tor10_int64 &rn);
    __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const tor10_int32 &rn);

    __device__ cuDoubleComplex operator*(const cuFloatComplex &ln, const cuDoubleComplex &rn);
    __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cuFloatComplex &rn);
    __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const tor10_double &rn);
    __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const tor10_float &rn);
    __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const tor10_uint64 &rn);
    __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const tor10_uint32 &rn);
    __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const tor10_int64 &rn);
    __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const tor10_int32 &rn);

    //__device__ cuDoubleComplex operator*(const cuDoubleComplex &rn,const cuDoubleComplex &ln);
    //__device__ cuDoubleComplex operator*(const cuFloatComplex &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator*(const tor10_double &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator*(const tor10_float &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator*(const tor10_uint64 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator*(const tor10_uint32 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator*(const tor10_int64 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator*(const tor10_int32 &rn,const cuDoubleComplex &ln);

    //__device__ cuDoubleComplex operator*(const cuDoubleComplex &rn,const cuFloatComplex &ln);
    //__device__ cuFloatComplex operator*(const cuFloatComplex &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator*(const tor10_double &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator*( const tor10_float &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/(const tor10_uint64 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/(const tor10_uint32 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/(const tor10_int64 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/( const tor10_int32 &rn,const cuFloatComplex &ln);

//----------------
    __device__ cuDoubleComplex operator/(const cuDoubleComplex &ln, const cuDoubleComplex &rn);
    __device__ cuDoubleComplex operator/(const cuDoubleComplex &ln, const cuFloatComplex &rn);
    __device__ cuDoubleComplex operator/(const cuDoubleComplex &ln, const tor10_double &rn);
    __device__ cuDoubleComplex operator/(const cuDoubleComplex &ln, const tor10_float &rn);
    __device__ cuDoubleComplex operator/(const cuDoubleComplex &ln, const tor10_uint64 &rn);
    __device__ cuDoubleComplex operator/(const cuDoubleComplex &ln, const tor10_uint32 &rn);
    __device__ cuDoubleComplex operator/(const cuDoubleComplex &ln, const tor10_int64 &rn);
    __device__ cuDoubleComplex operator/(const cuDoubleComplex &ln, const tor10_int32 &rn);

    __device__ cuDoubleComplex operator/(const cuFloatComplex &ln, const cuDoubleComplex &rn);
    __device__ cuFloatComplex operator/(const cuFloatComplex &ln, const cuFloatComplex &rn);
    __device__ cuFloatComplex operator/(const cuFloatComplex &ln, const tor10_double &rn);
    __device__ cuFloatComplex operator/(const cuFloatComplex &ln, const tor10_float &rn);
    __device__ cuFloatComplex operator/(const cuFloatComplex &ln, const tor10_uint64 &rn);
    __device__ cuFloatComplex operator/(const cuFloatComplex &ln, const tor10_uint32 &rn);
    __device__ cuFloatComplex operator/(const cuFloatComplex &ln, const tor10_int64 &rn);
    __device__ cuFloatComplex operator/(const cuFloatComplex &ln, const tor10_int32 &rn);

    //__device__ cuDoubleComplex operator/(const cuDoubleComplex &rn,const cuDoubleComplex &ln);
    //__device__ cuDoubleComplex operator/(const cuFloatComplex &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator/(const tor10_double &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator/(const tor10_float &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator/(const tor10_uint64 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator/(const tor10_uint32 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator/(const tor10_int64 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator/(const tor10_int32 &rn,const cuDoubleComplex &ln);

    //__device__ cuDoubleComplex operator/(const cuDoubleComplex &rn,const cuFloatComplex &ln);
    //__device__ cuFloatComplex operator/(const cuFloatComplex &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/(const tor10_double &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/( const tor10_float &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/(const tor10_uint64 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/(const tor10_uint32 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/(const tor10_int64 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/( const tor10_int32 &rn,const cuFloatComplex &ln);

#endif
}// namespace tor10
*/

#endif
