#ifndef __complex_arithmetic__H_
#define __complex_arithmetic__H_

#include "Type.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
#ifdef UNI_ICPC

#else
  // operator overload for CPU code. for COMPLEX type arithmic.
  // cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_complex128 &rn);
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_complex64 &rn);
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_double &rn);
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_float &rn);
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_uint64 &rn);
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_uint32 &rn);
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_int64 &rn);
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_int32 &rn);
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_uint16 &rn);
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_int16 &rn);
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_uint16 &rn);
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_bool &rn);

  cytnx_complex128 operator+(const cytnx_complex64 &ln, const cytnx_complex128 &rn);
  // cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_complex64 &rn);
  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_double &rn);
  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_float &rn);
  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_uint64 &rn);
  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_uint32 &rn);
  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_int64 &rn);
  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_int32 &rn);
  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_int16 &rn);
  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_uint16 &rn);
  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_bool &rn);

  // cytnx_complex128 operator+(const cytnx_complex128 &rn,const cytnx_complex128 &ln);
  // cytnx_complex128 operator+(const cytnx_complex64 &rn,const cytnx_complex128 &ln);
  cytnx_complex128 operator+(const cytnx_double &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator+(const cytnx_float &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator+(const cytnx_uint64 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator+(const cytnx_uint32 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator+(const cytnx_int64 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator+(const cytnx_int32 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator+(const cytnx_int16 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator+(const cytnx_uint16 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator+(const cytnx_bool &rn, const cytnx_complex128 &ln);

  // cytnx_complex128 operator+(const cytnx_complex128 &rn,const cytnx_complex64 &ln);
  // cytnx_complex64 operator+(const cytnx_complex64 &rn,const cytnx_complex64 &ln);
  cytnx_complex64 operator+(const cytnx_double &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator+(const cytnx_float &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator+(const cytnx_uint64 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator+(const cytnx_uint32 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator+(const cytnx_int64 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator+(const cytnx_int32 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator+(const cytnx_int16 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator+(const cytnx_uint16 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator+(const cytnx_bool &rn, const cytnx_complex64 &ln);

  //---------------------
  // cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_complex128 &rn);
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_complex64 &rn);
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_double &rn);
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_float &rn);
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_uint64 &rn);
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_uint32 &rn);
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_int64 &rn);
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_int32 &rn);
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_int16 &rn);
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_uint16 &rn);
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_bool &rn);

  cytnx_complex128 operator-(const cytnx_complex64 &ln, const cytnx_complex128 &rn);
  // cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_complex64 &rn);
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_double &rn);
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_float &rn);
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_uint64 &rn);
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_uint32 &rn);
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_int64 &rn);
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_int32 &rn);
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_uint16 &rn);
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_int16 &rn);
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_bool &rn);

  // cytnx_complex128 operator-(const cytnx_complex128 &rn,const cytnx_complex128 &ln);
  // cytnx_complex128 operator-(const cytnx_complex64 &rn,const cytnx_complex128 &ln);
  cytnx_complex128 operator-(const cytnx_double &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator-(const cytnx_float &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator-(const cytnx_uint64 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator-(const cytnx_uint32 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator-(const cytnx_int64 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator-(const cytnx_int32 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator-(const cytnx_uint16 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator-(const cytnx_int16 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator-(const cytnx_bool &rn, const cytnx_complex128 &ln);

  // cytnx_complex128 operator-(const cytnx_complex128 &rn,const cytnx_complex64 &ln);
  // cytnx_complex64 operator-(const cytnx_complex64 &rn,const cytnx_complex64 &ln);
  cytnx_complex64 operator-(const cytnx_double &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator-(const cytnx_float &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator-(const cytnx_uint64 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator-(const cytnx_uint32 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator-(const cytnx_int64 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator-(const cytnx_int32 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator-(const cytnx_uint16 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator-(const cytnx_int16 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator-(const cytnx_bool &rn, const cytnx_complex64 &ln);

  //---------------------
  // cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_complex128 &rn);
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_complex64 &rn);
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_double &rn);
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_float &rn);
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_uint64 &rn);
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_uint32 &rn);
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_int64 &rn);
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_int32 &rn);
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_uint16 &rn);
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_int16 &rn);
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_bool &rn);

  cytnx_complex128 operator*(const cytnx_complex64 &ln, const cytnx_complex128 &rn);
  // cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_complex64 &rn);
  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_double &rn);
  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_float &rn);
  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_uint64 &rn);
  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_uint32 &rn);
  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_int64 &rn);
  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_int32 &rn);
  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_uint16 &rn);
  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_int16 &rn);
  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_bool &rn);

  // cytnx_complex128 operator*(const cytnx_complex128 &rn,const cytnx_complex128 &ln);
  // cytnx_complex128 operator*(const cytnx_complex64 &rn,const cytnx_complex128 &ln);
  cytnx_complex128 operator*(const cytnx_double &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator*(const cytnx_float &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator*(const cytnx_uint64 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator*(const cytnx_uint32 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator*(const cytnx_int64 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator*(const cytnx_int32 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator*(const cytnx_uint16 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator*(const cytnx_int16 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator*(const cytnx_bool &rn, const cytnx_complex128 &ln);

  // cytnx_complex128 operator*(const cytnx_complex128 &rn,const cytnx_complex64 &ln);
  // cytnx_complex64 operator*(const cytnx_complex64 &rn,const cytnx_complex64 &ln);
  cytnx_complex64 operator*(const cytnx_double &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator*(const cytnx_float &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator*(const cytnx_uint64 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator*(const cytnx_uint32 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator*(const cytnx_int64 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator*(const cytnx_int32 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator*(const cytnx_uint16 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator*(const cytnx_int16 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator*(const cytnx_bool &rn, const cytnx_complex64 &ln);

  //---------------------
  // cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_complex128 &rn);
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_complex64 &rn);
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_double &rn);
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_float &rn);
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_uint64 &rn);
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_uint32 &rn);
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_int64 &rn);
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_int32 &rn);
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_uint16 &rn);
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_int16 &rn);
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_bool &rn);

  cytnx_complex128 operator/(const cytnx_complex64 &ln, const cytnx_complex128 &rn);
  // cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_complex64 &rn);
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_double &rn);
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_float &rn);
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_uint64 &rn);
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_uint32 &rn);
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_int64 &rn);
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_int32 &rn);
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_uint16 &rn);
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_int16 &rn);
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_bool &rn);

  // cytnx_complex128 operator/(const cytnx_complex128 &rn,const cytnx_complex128 &ln);
  // cytnx_complex128 operator/(const cytnx_complex64 &rn,const cytnx_complex128 &ln);
  cytnx_complex128 operator/(const cytnx_double &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator/(const cytnx_float &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator/(const cytnx_uint64 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator/(const cytnx_uint32 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator/(const cytnx_int64 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator/(const cytnx_int32 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator/(const cytnx_uint16 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator/(const cytnx_int16 &rn, const cytnx_complex128 &ln);
  cytnx_complex128 operator/(const cytnx_bool &rn, const cytnx_complex128 &ln);

  // cytnx_complex128 operator/(const cytnx_complex128 &rn,const cytnx_complex64 &ln);
  // cytnx_complex64 operator/(const cytnx_complex64 &rn,const cytnx_complex64 &ln);
  cytnx_complex64 operator/(const cytnx_double &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator/(const cytnx_float &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator/(const cytnx_uint64 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator/(const cytnx_uint32 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator/(const cytnx_int64 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator/(const cytnx_int32 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator/(const cytnx_uint16 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator/(const cytnx_int16 &rn, const cytnx_complex64 &ln);
  cytnx_complex64 operator/(const cytnx_bool &rn, const cytnx_complex64 &ln);

  // bool operator==(const cytnx_complex128 &ln, const cytnx_complex128 &rn);
  bool operator==(const cytnx_complex128 &ln, const cytnx_complex64 &rn);
  bool operator==(const cytnx_complex128 &ln, const cytnx_double &rn);
  bool operator==(const cytnx_complex128 &ln, const cytnx_float &rn);
  bool operator==(const cytnx_complex128 &ln, const cytnx_uint64 &rn);
  bool operator==(const cytnx_complex128 &ln, const cytnx_uint32 &rn);
  bool operator==(const cytnx_complex128 &ln, const cytnx_int64 &rn);
  bool operator==(const cytnx_complex128 &ln, const cytnx_int32 &rn);
  bool operator==(const cytnx_complex128 &ln, const cytnx_uint16 &rn);
  bool operator==(const cytnx_complex128 &ln, const cytnx_int16 &rn);
  bool operator==(const cytnx_complex128 &ln, const cytnx_uint16 &rn);
  bool operator==(const cytnx_complex128 &ln, const cytnx_bool &rn);

  bool operator==(const cytnx_complex64 &ln, const cytnx_complex128 &rn);
  // bool operator==(const cytnx_complex64 &ln, const cytnx_complex64 &rn);
  bool operator==(const cytnx_complex64 &ln, const cytnx_double &rn);
  bool operator==(const cytnx_complex64 &ln, const cytnx_float &rn);
  bool operator==(const cytnx_complex64 &ln, const cytnx_uint64 &rn);
  bool operator==(const cytnx_complex64 &ln, const cytnx_uint32 &rn);
  bool operator==(const cytnx_complex64 &ln, const cytnx_int64 &rn);
  bool operator==(const cytnx_complex64 &ln, const cytnx_int32 &rn);
  bool operator==(const cytnx_complex64 &ln, const cytnx_int16 &rn);
  bool operator==(const cytnx_complex64 &ln, const cytnx_uint16 &rn);
  bool operator==(const cytnx_complex64 &ln, const cytnx_bool &rn);

  // bool operator==(const cytnx_complex128 &rn,const cytnx_complex128 &ln);
  // bool operator==(const cytnx_complex64 &rn,const cytnx_complex128 &ln);
  bool operator==(const cytnx_double &rn, const cytnx_complex128 &ln);
  bool operator==(const cytnx_float &rn, const cytnx_complex128 &ln);
  bool operator==(const cytnx_uint64 &rn, const cytnx_complex128 &ln);
  bool operator==(const cytnx_uint32 &rn, const cytnx_complex128 &ln);
  bool operator==(const cytnx_int64 &rn, const cytnx_complex128 &ln);
  bool operator==(const cytnx_int32 &rn, const cytnx_complex128 &ln);
  bool operator==(const cytnx_int16 &rn, const cytnx_complex128 &ln);
  bool operator==(const cytnx_uint16 &rn, const cytnx_complex128 &ln);
  bool operator==(const cytnx_bool &rn, const cytnx_complex128 &ln);

  // bool operator==(const cytnx_complex128 &rn,const cytnx_complex64 &ln);
  // bool operator==(const cytnx_complex64 &rn,const cytnx_complex64 &ln);
  bool operator==(const cytnx_double &rn, const cytnx_complex64 &ln);
  bool operator==(const cytnx_float &rn, const cytnx_complex64 &ln);
  bool operator==(const cytnx_uint64 &rn, const cytnx_complex64 &ln);
  bool operator==(const cytnx_uint32 &rn, const cytnx_complex64 &ln);
  bool operator==(const cytnx_int64 &rn, const cytnx_complex64 &ln);
  bool operator==(const cytnx_int32 &rn, const cytnx_complex64 &ln);
  bool operator==(const cytnx_int16 &rn, const cytnx_complex64 &ln);
  bool operator==(const cytnx_uint16 &rn, const cytnx_complex64 &ln);
  bool operator==(const cytnx_bool &rn, const cytnx_complex64 &ln);
#endif

#if defined(UNI_GPU)
  //__host__ __device__ bool operator==(const cuDoubleComplex &ln, const cuDoubleComplex &rn);
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cuFloatComplex &rn);
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_double &rn);
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_float &rn);
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_uint64 &rn);
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_uint32 &rn);
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_int64 &rn);
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_int32 &rn);
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_uint16 &rn);
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_int16 &rn);
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_uint16 &rn);
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_bool &rn);

  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cuDoubleComplex &rn);
  //__host__ __device__ bool operator==(const cuFloatComplex &ln, const cuFloatComplex &rn);
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_double &rn);
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_float &rn);
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_uint64 &rn);
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_uint32 &rn);
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_int64 &rn);
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_int32 &rn);
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_int16 &rn);
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_uint16 &rn);
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_bool &rn);

  //__host__ __device__ bool operator==(const cuDoubleComplex &rn,const cuDoubleComplex &ln);
  //__host__ __device__ bool operator==(const cuFloatComplex &rn,const cuDoubleComplex &ln);
  __host__ __device__ bool operator==(const cytnx_double &rn, const cuDoubleComplex &ln);
  __host__ __device__ bool operator==(const cytnx_float &rn, const cuDoubleComplex &ln);
  __host__ __device__ bool operator==(const cytnx_uint64 &rn, const cuDoubleComplex &ln);
  __host__ __device__ bool operator==(const cytnx_uint32 &rn, const cuDoubleComplex &ln);
  __host__ __device__ bool operator==(const cytnx_int64 &rn, const cuDoubleComplex &ln);
  __host__ __device__ bool operator==(const cytnx_int32 &rn, const cuDoubleComplex &ln);
  __host__ __device__ bool operator==(const cytnx_int16 &rn, const cuDoubleComplex &ln);
  __host__ __device__ bool operator==(const cytnx_uint16 &rn, const cuDoubleComplex &ln);
  __host__ __device__ bool operator==(const cytnx_bool &rn, const cuDoubleComplex &ln);

  //__host__ __device__ bool operator==(const cuDoubleComplex &rn,const cuFloatComplex &ln);
  //__host__ __device__ bool operator==(const cuFloatComplex &rn,const cuFloatComplex &ln);
  __host__ __device__ bool operator==(const cytnx_double &rn, const cuFloatComplex &ln);
  __host__ __device__ bool operator==(const cytnx_float &rn, const cuFloatComplex &ln);
  __host__ __device__ bool operator==(const cytnx_uint64 &rn, const cuFloatComplex &ln);
  __host__ __device__ bool operator==(const cytnx_uint32 &rn, const cuFloatComplex &ln);
  __host__ __device__ bool operator==(const cytnx_int64 &rn, const cuFloatComplex &ln);
  __host__ __device__ bool operator==(const cytnx_int32 &rn, const cuFloatComplex &ln);
  __host__ __device__ bool operator==(const cytnx_int16 &rn, const cuFloatComplex &ln);
  __host__ __device__ bool operator==(const cytnx_uint16 &rn, const cuFloatComplex &ln);
  __host__ __device__ bool operator==(const cytnx_bool &rn, const cuFloatComplex &ln);
#endif

  // operator overload for GPU code. for COMPLEX type arithmic.
}  // namespace cytnx

#endif
