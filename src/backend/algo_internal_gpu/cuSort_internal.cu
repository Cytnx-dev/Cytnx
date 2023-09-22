#include "cuSort_internal.hpp"
#include "backend/algo_internal_interface.hpp"
#include <iostream>
#include <algorithm>
#include <thrust/sort.h>

namespace cytnx {

  namespace algo_internal {

    bool cu_compare_c128(cytnx_complex128 a, cytnx_complex128 b) {
      if (real(a) == real(b)) return imag(a) < imag(b);
      return real(a) < real(b);
    }
    void cuSort_internal_cd(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                            const cytnx_uint64 &Nelem) {
      cytnx_complex128 *p = (cytnx_complex128 *)out->Mem;
      cytnx_uint64 Niter = Nelem / stride;
      for (cytnx_uint64 i = 0; i < Niter; i++)
        thrust::sort(p + i * stride, p + i * stride + stride, cu_compare_c128);
    }
    bool cu_compare_c64(cytnx_complex64 a, cytnx_complex64 b) {
      if (real(a) == real(b)) return imag(a) < imag(b);
      return real(a) < real(b);
    }
    void cuSort_internal_cf(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                            const cytnx_uint64 &Nelem) {
      cytnx_complex64 *p = (cytnx_complex64 *)out->Mem;
      cytnx_uint64 Niter = Nelem / stride;
      for (cytnx_uint64 i = 0; i < Niter; i++)
        thrust::sort(p + i * stride, p + i * stride + stride, cu_compare_c64);
    }

    void cuSort_internal_d(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                           const cytnx_uint64 &Nelem) {
      double *p = (double *)out->Mem;
      cytnx_uint64 Niter = Nelem / stride;
      for (cytnx_uint64 i = 0; i < Niter; i++)
        thrust::sort(p + i * stride, p + i * stride + stride);
    }

    void cuSort_internal_f(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                           const cytnx_uint64 &Nelem) {
      float *p = (float *)out->Mem;
      cytnx_uint64 Niter = Nelem / stride;
      for (cytnx_uint64 i = 0; i < Niter; i++)
        thrust::sort(p + i * stride, p + i * stride + stride);
    }

    void cuSort_internal_u64(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                             const cytnx_uint64 &Nelem) {
      cytnx_uint64 *p = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 Niter = Nelem / stride;
      for (cytnx_uint64 i = 0; i < Niter; i++)
        thrust::sort(p + i * stride, p + i * stride + stride);
    }

    void cuSort_internal_i64(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                             const cytnx_uint64 &Nelem) {
      cytnx_int64 *p = (cytnx_int64 *)out->Mem;
      cytnx_uint64 Niter = Nelem / stride;
      for (cytnx_uint64 i = 0; i < Niter; i++)
        thrust::sort(p + i * stride, p + i * stride + stride);
    }

    void cuSort_internal_u32(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                             const cytnx_uint64 &Nelem) {
      cytnx_uint32 *p = (cytnx_uint32 *)out->Mem;
      cytnx_uint64 Niter = Nelem / stride;
      for (cytnx_uint64 i = 0; i < Niter; i++)
        thrust::sort(p + i * stride, p + i * stride + stride);
    }

    void cuSort_internal_i32(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                             const cytnx_uint64 &Nelem) {
      cytnx_int32 *p = (cytnx_int32 *)out->Mem;
      cytnx_uint64 Niter = Nelem / stride;
      for (cytnx_uint64 i = 0; i < Niter; i++)
        thrust::sort(p + i * stride, p + i * stride + stride);
    }

    void cuSort_internal_u16(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                             const cytnx_uint64 &Nelem) {
      cytnx_uint16 *p = (cytnx_uint16 *)out->Mem;
      cytnx_uint64 Niter = Nelem / stride;
      for (cytnx_uint64 i = 0; i < Niter; i++)
        thrust::sort(p + i * stride, p + i * stride + stride);
    }

    void cuSort_internal_i16(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                             const cytnx_uint64 &Nelem) {
      cytnx_int16 *p = (cytnx_int16 *)out->Mem;
      cytnx_uint64 Niter = Nelem / stride;
      for (cytnx_uint64 i = 0; i < Niter; i++)
        thrust::sort(p + i * stride, p + i * stride + stride);
    }

    void cuSort_internal_b(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                           const cytnx_uint64 &Nelem) {
      /*
      cytnx_bool *p = (cytnx_bool*)out->Mem;
      cytnx_uint64 Niter = Nelem/stride;
      for(cytnx_uint64 i=0;i<Niter;i++)
          std::sort(p+i*stride,p+i*stride+stride);
      */
      cytnx_error_msg(true, "[ERROR] cytnx currently does not have bool type sort.%s", "\n");
    }

  }  // namespace algo_internal

}  // namespace cytnx
