#include "Type.hpp"
#include "utils/cucomplex_arithmetic.hpp"
namespace cytnx {

#ifdef UNI_GPU

  //-----------------------------
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cuDoubleComplex &rn) {
    return (rn.x == ln.x) && (rn.y == ln.y);
  }
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cuFloatComplex &rn) {
    return ln == cuComplexFloatToDouble(rn);
  }
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_double &rn) {
    return ln == make_cuDoubleComplex(rn, 0);
  }
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_float &rn) {
    return ln == make_cuDoubleComplex(rn, 0);
  }
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_uint64 &rn) {
    return ln == make_cuDoubleComplex(rn, 0);
  }
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_uint32 &rn) {
    return ln == make_cuDoubleComplex(rn, 0);
  }
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_int64 &rn) {
    return ln == make_cuDoubleComplex(rn, 0);
  }
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_int32 &rn) {
    return ln == make_cuDoubleComplex(rn, 0);
  }
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_uint16 &rn) {
    return ln == make_cuDoubleComplex(rn, 0);
  }
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_int16 &rn) {
    return ln == make_cuDoubleComplex(rn, 0);
  }
  __host__ __device__ bool operator==(const cuDoubleComplex &ln, const cytnx_bool &rn) {
    return ln == make_cuDoubleComplex(rn, 0);
  }

  //-----------------------------
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cuFloatComplex &rn) {
    return (ln.x == rn.x) && (ln.y == rn.y);
  }
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cuDoubleComplex &rn) {
    return cuComplexFloatToDouble(ln) == rn;
  }

  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_double &rn) {
    return make_cuFloatComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_float &rn) {
    return make_cuFloatComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_uint64 &rn) {
    return make_cuFloatComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_uint32 &rn) {
    return make_cuFloatComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_int64 &rn) {
    return make_cuFloatComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_int32 &rn) {
    return make_cuFloatComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_uint16 &rn) {
    return make_cuFloatComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_int16 &rn) {
    return make_cuFloatComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cuFloatComplex &ln, const cytnx_bool &rn) {
    return make_cuFloatComplex(rn, 0) == ln;
  }
  //-----------------------
  //__host__ __device__ bool operator==(const cuDoubleComplex &rn,const cuDoubleComplex &ln);
  //__host__ __device__ bool operator==(const cuFloatComplex &rn,const cuDoubleComplex &ln);
  __host__ __device__ bool operator==(const cytnx_double &rn, const cuDoubleComplex &ln) {
    return make_cuDoubleComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cytnx_float &rn, const cuDoubleComplex &ln) {
    return make_cuDoubleComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cytnx_uint64 &rn, const cuDoubleComplex &ln) {
    return make_cuDoubleComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cytnx_uint32 &rn, const cuDoubleComplex &ln) {
    return make_cuDoubleComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cytnx_int64 &rn, const cuDoubleComplex &ln) {
    return make_cuDoubleComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cytnx_int32 &rn, const cuDoubleComplex &ln) {
    return make_cuDoubleComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cytnx_uint16 &rn, const cuDoubleComplex &ln) {
    return make_cuDoubleComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cytnx_int16 &rn, const cuDoubleComplex &ln) {
    return make_cuDoubleComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cytnx_bool &rn, const cuDoubleComplex &ln) {
    return make_cuDoubleComplex(rn, 0) == ln;
  }
  //----------------------

  //__host__ __device__ bool operator==(const cuDoubleComplex &rn,const cuFloatComplex &ln);
  //__host__ __device__ bool operator==(const cuFloatComplex &rn,const cuFloatComplex &ln);
  __host__ __device__ bool operator==(const cytnx_double &rn, const cuFloatComplex &ln) {
    return make_cuFloatComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cytnx_float &rn, const cuFloatComplex &ln) {
    return make_cuFloatComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cytnx_uint64 &rn, const cuFloatComplex &ln) {
    return make_cuFloatComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cytnx_uint32 &rn, const cuFloatComplex &ln) {
    return make_cuFloatComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cytnx_int64 &rn, const cuFloatComplex &ln) {
    return make_cuFloatComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cytnx_int32 &rn, const cuFloatComplex &ln) {
    return make_cuFloatComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cytnx_uint16 &rn, const cuFloatComplex &ln) {
    return make_cuFloatComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cytnx_int16 &rn, const cuFloatComplex &ln) {
    return make_cuFloatComplex(rn, 0) == ln;
  }
  __host__ __device__ bool operator==(const cytnx_bool &rn, const cuFloatComplex &ln) {
    return make_cuFloatComplex(rn, 0) == ln;
  }

  //[MUL]

  //-----------------------------
  __host__ __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln,
                                                const cuDoubleComplex &rn) {
    return cuCmul(ln, rn);
  }
  __host__ __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln,
                                                const cuFloatComplex &rn) {
    return cuCmul(ln, cuComplexFloatToDouble(rn));
  }
  __host__ __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cytnx_double &rn) {
    return cuCmul(ln, make_cuDoubleComplex(rn, 0));
  }
  __host__ __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cytnx_float &rn) {
    return cuCmul(ln, make_cuDoubleComplex(rn, 0));
  }
  __host__ __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cytnx_uint64 &rn) {
    return cuCmul(ln, make_cuDoubleComplex(rn, 0));
  }
  __host__ __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cytnx_uint32 &rn) {
    return cuCmul(ln, make_cuDoubleComplex(rn, 0));
  }
  __host__ __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cytnx_int64 &rn) {
    return cuCmul(ln, make_cuDoubleComplex(rn, 0));
  }
  __host__ __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cytnx_int32 &rn) {
    return cuCmul(ln, make_cuDoubleComplex(rn, 0));
  }
  __host__ __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cytnx_uint16 &rn) {
    return cuCmul(ln, make_cuDoubleComplex(rn, 0));
  }
  __host__ __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cytnx_int16 &rn) {
    return cuCmul(ln, make_cuDoubleComplex(rn, 0));
  }
  __host__ __device__ cuDoubleComplex operator*(const cuDoubleComplex &ln, const cytnx_bool &rn) {
    return cuCmul(ln, make_cuDoubleComplex(rn, 0));
  }

  //-----------------------------
  __host__ __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cuFloatComplex &rn) {
    return cuCmulf(ln, rn);
  }
  __host__ __device__ cuDoubleComplex operator*(const cuFloatComplex &ln,
                                                const cuDoubleComplex &rn) {
    return cuCmul(cuComplexFloatToDouble(ln), rn);
  }

  __host__ __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cytnx_double &rn) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }
  __host__ __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cytnx_float &rn) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }
  __host__ __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cytnx_uint64 &rn) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }
  __host__ __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cytnx_uint32 &rn) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }
  __host__ __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cytnx_int64 &rn) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }
  __host__ __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cytnx_int32 &rn) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }
  __host__ __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cytnx_uint16 &rn) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }
  __host__ __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cytnx_int16 &rn) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }
  __host__ __device__ cuFloatComplex operator*(const cuFloatComplex &ln, const cytnx_bool &rn) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }
  //-----------------------
  __host__ __device__ cuDoubleComplex operator*(const cytnx_double &rn, const cuDoubleComplex &ln) {
    return cuCmul(make_cuDoubleComplex(rn, 0), ln);
  }
  __host__ __device__ cuDoubleComplex operator*(const cytnx_float &rn, const cuDoubleComplex &ln) {
    return cuCmul(make_cuDoubleComplex(rn, 0), ln);
  }
  __host__ __device__ cuDoubleComplex operator*(const cytnx_uint64 &rn, const cuDoubleComplex &ln) {
    return cuCmul(make_cuDoubleComplex(rn, 0), ln);
  }
  __host__ __device__ cuDoubleComplex operator*(const cytnx_uint32 &rn, const cuDoubleComplex &ln) {
    return cuCmul(make_cuDoubleComplex(rn, 0), ln);
  }
  __host__ __device__ cuDoubleComplex operator*(const cytnx_int64 &rn, const cuDoubleComplex &ln) {
    return cuCmul(make_cuDoubleComplex(rn, 0), ln);
  }
  __host__ __device__ cuDoubleComplex operator*(const cytnx_int32 &rn, const cuDoubleComplex &ln) {
    return cuCmul(make_cuDoubleComplex(rn, 0), ln);
  }
  __host__ __device__ cuDoubleComplex operator*(const cytnx_uint16 &rn, const cuDoubleComplex &ln) {
    return cuCmul(make_cuDoubleComplex(rn, 0), ln);
  }
  __host__ __device__ cuDoubleComplex operator*(const cytnx_int16 &rn, const cuDoubleComplex &ln) {
    return cuCmul(make_cuDoubleComplex(rn, 0), ln);
  }
  __host__ __device__ cuDoubleComplex operator*(const bool &rn, const cuDoubleComplex &ln) {
    return cuCmul(make_cuDoubleComplex(rn, 0), ln);
  }
  //----------------------

  __host__ __device__ cuFloatComplex operator*(const cytnx_double &rn, const cuFloatComplex &ln) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }
  __host__ __device__ cuFloatComplex operator*(const cytnx_float &rn, const cuFloatComplex &ln) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }
  __host__ __device__ cuFloatComplex operator*(const cytnx_uint64 &rn, const cuFloatComplex &ln) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }
  __host__ __device__ cuFloatComplex operator*(const cytnx_uint32 &rn, const cuFloatComplex &ln) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }
  __host__ __device__ cuFloatComplex operator*(const cytnx_int64 &rn, const cuFloatComplex &ln) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }
  __host__ __device__ cuFloatComplex operator*(const cytnx_int32 &rn, const cuFloatComplex &ln) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }
  __host__ __device__ cuFloatComplex operator*(const cytnx_uint16 &rn, const cuFloatComplex &ln) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }
  __host__ __device__ cuFloatComplex operator*(const cytnx_int16 &rn, const cuFloatComplex &ln) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }
  __host__ __device__ cuFloatComplex operator*(const cytnx_bool &rn, const cuFloatComplex &ln) {
    return cuCmulf(make_cuFloatComplex(rn, 0), ln);
  }

#endif

}  // namespace cytnx
