#ifndef __complex_arithmic__H_
#define __complex_arithmic__H_

#include "Type.hpp"
#include "cytnx_error.hpp"


namespace cytnx{

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


// operator overload for GPU code. for COMPLEX type arithmic.
} // namespace cytnx


// device complex op. 
/*
namespace cytnx{

#ifdef UNI_GPU
    __device__ cuDoubleComplex operator+(const cuDoubleComplex &ln, const cuDoubleComplex &rn);
    __device__ cuDoubleComplex operator+(const cuDoubleComplex &ln, const cuFloatComplex &rn);
    __device__ cuDoubleComplex operator+(const cuDoubleComplex &ln, const cytnx_double &rn);
    __device__ cuDoubleComplex operator+(const cuDoubleComplex &ln, const cytnx_float &rn);
    __device__ cuDoubleComplex operator+(const cuDoubleComplex &ln, const cytnx_uint64 &rn);
    __device__ cuDoubleComplex operator+(const cuDoubleComplex &ln, const cytnx_uint32 &rn);
    __device__ cuDoubleComplex operator+(const cuDoubleComplex &ln, const cytnx_int64 &rn);
    __device__ cuDoubleComplex operator+(const cuDoubleComplex &ln, const cytnx_int32 &rn);

    __device__ cuDoubleComplex operator+(const cuFloatComplex &ln, const cuDoubleComplex &rn);
    __device__ cuFloatComplex operator+(const cuFloatComplex &ln, const cuFloatComplex &rn);
    __device__ cuFloatComplex operator+(const cuFloatComplex &ln, const cytnx_double &rn);
    __device__ cuFloatComplex operator+(const cuFloatComplex &ln, const cytnx_float &rn);
    __device__ cuFloatComplex operator+(const cuFloatComplex &ln, const cytnx_uint64 &rn);
    __device__ cuFloatComplex operator+(const cuFloatComplex &ln, const cytnx_uint32 &rn);
    __device__ cuFloatComplex operator+(const cuFloatComplex &ln, const cytnx_int64 &rn);
    __device__ cuFloatComplex operator+(const cuFloatComplex &ln, const cytnx_int32 &rn);

    //__device__ cuDoubleComplex operator+(const cuDoubleComplex &rn,const cuDoubleComplex &ln);
    //__device__ cuDoubleComplex operator+(const cuFloatComplex &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator+(const cytnx_double &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator+(const cytnx_float &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator+(const cytnx_uint64 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator+(const cytnx_uint32 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator+(const cytnx_int64 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator+(const cytnx_int32 &rn,const cuDoubleComplex &ln);

    //__device__ cuDoubleComplex operator+(const cuDoubleComplex &rn,const cuFloatComplex &ln);
    //__device__ cuFloatComplex operator+(const cuFloatComplex &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator+(const cytnx_double &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator+( const cytnx_float &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator+(const cytnx_uint64 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator+(const cytnx_uint32 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator+(const cytnx_int64 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator+( const cytnx_int32 &rn,const cuFloatComplex &ln);

//----------------
    __device__ cuDoubleComplex operator-(const cuDoubleComplex &ln, const cuDoubleComplex &rn);
    __device__ cuDoubleComplex operator-(const cuDoubleComplex &ln, const cuFloatComplex &rn);
    __device__ cuDoubleComplex operator-(const cuDoubleComplex &ln, const cytnx_double &rn);
    __device__ cuDoubleComplex operator-(const cuDoubleComplex &ln, const cytnx_float &rn);
    __device__ cuDoubleComplex operator-(const cuDoubleComplex &ln, const cytnx_uint64 &rn);
    __device__ cuDoubleComplex operator-(const cuDoubleComplex &ln, const cytnx_uint32 &rn);
    __device__ cuDoubleComplex operator-(const cuDoubleComplex &ln, const cytnx_int64 &rn);
    __device__ cuDoubleComplex operator-(const cuDoubleComplex &ln, const cytnx_int32 &rn);

    __device__ cuDoubleComplex operator-(const cuFloatComplex &ln, const cuDoubleComplex &rn);
    __device__ cuFloatComplex operator-(const cuFloatComplex &ln, const cuFloatComplex &rn);
    __device__ cuFloatComplex operator-(const cuFloatComplex &ln, const cytnx_double &rn);
    __device__ cuFloatComplex operator-(const cuFloatComplex &ln, const cytnx_float &rn);
    __device__ cuFloatComplex operator-(const cuFloatComplex &ln, const cytnx_uint64 &rn);
    __device__ cuFloatComplex operator-(const cuFloatComplex &ln, const cytnx_uint32 &rn);
    __device__ cuFloatComplex operator-(const cuFloatComplex &ln, const cytnx_int64 &rn);
    __device__ cuFloatComplex operator-(const cuFloatComplex &ln, const cytnx_int32 &rn);

    //__device__ cuDoubleComplex operator-(const cuDoubleComplex &rn,const cuDoubleComplex &ln);
    //__device__ cuDoubleComplex operator-(const cuFloatComplex &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator-(const cytnx_double &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator-(const cytnx_float &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator-(const cytnx_uint64 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator-(const cytnx_uint32 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator-(const cytnx_int64 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator-(const cytnx_int32 &rn,const cuDoubleComplex &ln);

    //__device__ cuDoubleComplex operator-(const cuDoubleComplex &rn,const cuFloatComplex &ln);
    //__device__ cuFloatComplex operator-(const cuFloatComplex &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator-(const cytnx_double &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator-( const cytnx_float &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator-(const cytnx_uint64 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator-(const cytnx_uint32 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator-(const cytnx_int64 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator-( const cytnx_int32 &rn,const cuFloatComplex &ln);

//----------------
    __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cuDoubleComplex &rn);
    __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cuFloatComplex &rn);
    __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cytnx_double &rn);
    __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cytnx_float &rn);
    __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cytnx_uint64 &rn);
    __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cytnx_uint32 &rn);
    __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cytnx_int64 &rn);
    __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cytnx_int32 &rn);

    __device__ cuDoubleComplex operator*(const cuFloatComplex &ln, const cuDoubleComplex &rn);
    __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cuFloatComplex &rn);
    __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cytnx_double &rn);
    __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cytnx_float &rn);
    __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cytnx_uint64 &rn);
    __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cytnx_uint32 &rn);
    __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cytnx_int64 &rn);
    __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cytnx_int32 &rn);

    //__device__ cuDoubleComplex operator*(const cuDoubleComplex &rn,const cuDoubleComplex &ln);
    //__device__ cuDoubleComplex operator*(const cuFloatComplex &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator*(const cytnx_double &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator*(const cytnx_float &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator*(const cytnx_uint64 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator*(const cytnx_uint32 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator*(const cytnx_int64 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator*(const cytnx_int32 &rn,const cuDoubleComplex &ln);

    //__device__ cuDoubleComplex operator*(const cuDoubleComplex &rn,const cuFloatComplex &ln);
    //__device__ cuFloatComplex operator*(const cuFloatComplex &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator*(const cytnx_double &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator*( const cytnx_float &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/(const cytnx_uint64 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/(const cytnx_uint32 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/(const cytnx_int64 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/( const cytnx_int32 &rn,const cuFloatComplex &ln);

//----------------
    __device__ cuDoubleComplex operator/(const cuDoubleComplex &ln, const cuDoubleComplex &rn);
    __device__ cuDoubleComplex operator/(const cuDoubleComplex &ln, const cuFloatComplex &rn);
    __device__ cuDoubleComplex operator/(const cuDoubleComplex &ln, const cytnx_double &rn);
    __device__ cuDoubleComplex operator/(const cuDoubleComplex &ln, const cytnx_float &rn);
    __device__ cuDoubleComplex operator/(const cuDoubleComplex &ln, const cytnx_uint64 &rn);
    __device__ cuDoubleComplex operator/(const cuDoubleComplex &ln, const cytnx_uint32 &rn);
    __device__ cuDoubleComplex operator/(const cuDoubleComplex &ln, const cytnx_int64 &rn);
    __device__ cuDoubleComplex operator/(const cuDoubleComplex &ln, const cytnx_int32 &rn);

    __device__ cuDoubleComplex operator/(const cuFloatComplex &ln, const cuDoubleComplex &rn);
    __device__ cuFloatComplex operator/(const cuFloatComplex &ln, const cuFloatComplex &rn);
    __device__ cuFloatComplex operator/(const cuFloatComplex &ln, const cytnx_double &rn);
    __device__ cuFloatComplex operator/(const cuFloatComplex &ln, const cytnx_float &rn);
    __device__ cuFloatComplex operator/(const cuFloatComplex &ln, const cytnx_uint64 &rn);
    __device__ cuFloatComplex operator/(const cuFloatComplex &ln, const cytnx_uint32 &rn);
    __device__ cuFloatComplex operator/(const cuFloatComplex &ln, const cytnx_int64 &rn);
    __device__ cuFloatComplex operator/(const cuFloatComplex &ln, const cytnx_int32 &rn);

    //__device__ cuDoubleComplex operator/(const cuDoubleComplex &rn,const cuDoubleComplex &ln);
    //__device__ cuDoubleComplex operator/(const cuFloatComplex &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator/(const cytnx_double &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator/(const cytnx_float &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator/(const cytnx_uint64 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator/(const cytnx_uint32 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator/(const cytnx_int64 &rn,const cuDoubleComplex &ln);
    __device__ cuDoubleComplex operator/(const cytnx_int32 &rn,const cuDoubleComplex &ln);

    //__device__ cuDoubleComplex operator/(const cuDoubleComplex &rn,const cuFloatComplex &ln);
    //__device__ cuFloatComplex operator/(const cuFloatComplex &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/(const cytnx_double &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/( const cytnx_float &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/(const cytnx_uint64 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/(const cytnx_uint32 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/(const cytnx_int64 &rn,const cuFloatComplex &ln);
    __device__ cuFloatComplex operator/( const cytnx_int32 &rn,const cuFloatComplex &ln);

#endif
}// namespace cytnx
*/

#endif
