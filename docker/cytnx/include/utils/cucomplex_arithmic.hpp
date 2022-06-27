#ifndef __cucomplex_arithmic__H_
#define __cucomplex_arithmic__H_

#include "Type.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
#ifdef UNI_GPU
  //__host__ __device__ bool operator==(const cuDoubleComplex &ln, const cuDoubleComplex &rn){
  //    return (ln.x == rn.x)&&(ln.y == rn.y);
  //};
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cuDoubleComplex &rn);
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
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cuFloatComplex &rn);
  //__host__ __device__ bool operator==(const cuFloatComplex &ln, const cuFloatComplex &rn){
  //    return (ln.x == rn.x)&&(ln.y == rn.y);
  //};
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
