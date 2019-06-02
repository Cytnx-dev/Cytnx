#ifndef __complex_arithmic__H_
#define __complex_arithmic__H_

#include "Type.hpp"
#include "cytnx_error.hpp"


namespace cytnx{
#ifdef UNI_MKL

#else
// operator overload for CPU code. for COMPLEX type arithmic.
    //cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_complex128 &rn);
    cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_complex64 &rn);
    cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_double &rn);
    cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_float &rn);
    cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_uint64 &rn);
    cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_uint32 &rn);
    cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_int64 &rn);
    cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_int32 &rn);

    cytnx_complex128 operator+(const cytnx_complex64 &ln, const cytnx_complex128 &rn);
    //cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_complex64 &rn);
    cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_double &rn);
    cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_float &rn);
    cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_uint64 &rn);
    cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_uint32 &rn);
    cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_int64 &rn);
    cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_int32 &rn);

    //cytnx_complex128 operator+(const cytnx_complex128 &rn,const cytnx_complex128 &ln);
    //cytnx_complex128 operator+(const cytnx_complex64 &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator+(const cytnx_double &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator+(const cytnx_float &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator+(const cytnx_uint64 &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator+(const cytnx_uint32 &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator+(const cytnx_int64 &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator+(const cytnx_int32 &rn,const cytnx_complex128 &ln);

    //cytnx_complex128 operator+(const cytnx_complex128 &rn,const cytnx_complex64 &ln);
    //cytnx_complex64 operator+(const cytnx_complex64 &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator+(const cytnx_double &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator+( const cytnx_float &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator+(const cytnx_uint64 &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator+(const cytnx_uint32 &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator+(const cytnx_int64 &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator+( const cytnx_int32 &rn,const cytnx_complex64 &ln);

//---------------------
    //cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_complex128 &rn);
    cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_complex64 &rn);
    cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_double &rn);
    cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_float &rn);
    cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_uint64 &rn);
    cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_uint32 &rn);
    cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_int64 &rn);
    cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_int32 &rn);

    cytnx_complex128 operator-(const cytnx_complex64 &ln, const cytnx_complex128 &rn);
    //cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_complex64 &rn);
    cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_double &rn);
    cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_float &rn);
    cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_uint64 &rn);
    cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_uint32 &rn);
    cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_int64 &rn);
    cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_int32 &rn);

    //cytnx_complex128 operator-(const cytnx_complex128 &rn,const cytnx_complex128 &ln);
    //cytnx_complex128 operator-(const cytnx_complex64 &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator-(const cytnx_double &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator-(const cytnx_float &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator-(const cytnx_uint64 &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator-(const cytnx_uint32 &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator-(const cytnx_int64 &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator-(const cytnx_int32 &rn,const cytnx_complex128 &ln);

    //cytnx_complex128 operator-(const cytnx_complex128 &rn,const cytnx_complex64 &ln);
    //cytnx_complex64 operator-(const cytnx_complex64 &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator-(const cytnx_double &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator-( const cytnx_float &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator-(const cytnx_uint64 &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator-(const cytnx_uint32 &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator-(const cytnx_int64 &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator-( const cytnx_int32 &rn,const cytnx_complex64 &ln);

//---------------------
    //cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_complex128 &rn);
    cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_complex64 &rn);
    cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_double &rn);
    cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_float &rn);
    cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_uint64 &rn);
    cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_uint32 &rn);
    cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_int64 &rn);
    cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_int32 &rn);

    cytnx_complex128 operator*(const cytnx_complex64 &ln, const cytnx_complex128 &rn);
    //cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_complex64 &rn);
    cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_double &rn);
    cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_float &rn);
    cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_uint64 &rn);
    cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_uint32 &rn);
    cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_int64 &rn);
    cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_int32 &rn);

    //cytnx_complex128 operator*(const cytnx_complex128 &rn,const cytnx_complex128 &ln);
    //cytnx_complex128 operator*(const cytnx_complex64 &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator*(const cytnx_double &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator*(const cytnx_float &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator*(const cytnx_uint64 &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator*(const cytnx_uint32 &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator*(const cytnx_int64 &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator*(const cytnx_int32 &rn,const cytnx_complex128 &ln);

    //cytnx_complex128 operator*(const cytnx_complex128 &rn,const cytnx_complex64 &ln);
    //cytnx_complex64 operator*(const cytnx_complex64 &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator*(const cytnx_double &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator*( const cytnx_float &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator*(const cytnx_uint64 &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator*(const cytnx_uint32 &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator*(const cytnx_int64 &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator*( const cytnx_int32 &rn,const cytnx_complex64 &ln);

//---------------------
    //cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_complex128 &rn);
    cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_complex64 &rn);
    cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_double &rn);
    cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_float &rn);
    cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_uint64 &rn);
    cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_uint32 &rn);
    cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_int64 &rn);
    cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_int32 &rn);

    cytnx_complex128 operator/(const cytnx_complex64 &ln, const cytnx_complex128 &rn);
    //cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_complex64 &rn);
    cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_double &rn);
    cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_float &rn);
    cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_uint64 &rn);
    cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_uint32 &rn);
    cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_int64 &rn);
    cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_int32 &rn);

    //cytnx_complex128 operator/(const cytnx_complex128 &rn,const cytnx_complex128 &ln);
    //cytnx_complex128 operator/(const cytnx_complex64 &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator/(const cytnx_double &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator/(const cytnx_float &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator/(const cytnx_uint64 &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator/(const cytnx_uint32 &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator/(const cytnx_int64 &rn,const cytnx_complex128 &ln);
    cytnx_complex128 operator/(const cytnx_int32 &rn,const cytnx_complex128 &ln);

    //cytnx_complex128 operator/(const cytnx_complex128 &rn,const cytnx_complex64 &ln);
    //cytnx_complex64 operator/(const cytnx_complex64 &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator/(const cytnx_double &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator/( const cytnx_float &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator/(const cytnx_uint64 &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator/(const cytnx_uint32 &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator/(const cytnx_int64 &rn,const cytnx_complex64 &ln);
    cytnx_complex64 operator/( const cytnx_int32 &rn,const cytnx_complex64 &ln);
#endif

// operator overload for GPU code. for COMPLEX type arithmic.
} // namespace cytnx

#endif
