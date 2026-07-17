#include "cuSetArange_gpu.hpp"

namespace cytnx {
  namespace utils_internal {

    // The linear index is computed once in 64-bit: `blockIdx.x * blockDim.x` is 32-bit and
    // overflows past 2^32 elements. The value is `start + step * idx` -- the earlier generic
    // kernel dropped the parentheses (`start + step * blk * dim + thread`), so the intra-block
    // offset was added un-scaled by step and non-unit steps were ignored (#1070/#1076).
    template <class T>
    __global__ void cuSetArange_kernel(T *in, cytnx_double start, cytnx_double step,
                                       cytnx_uint64 Nelem) {
      const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      if (idx < Nelem) {
        in[idx] = start + step * idx;
      }
    }
    __global__ void cuSetArange_kernel(cuDoubleComplex *in, cytnx_double start, cytnx_double step,
                                       cytnx_uint64 Nelem) {
      const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      if (idx < Nelem) {
        in[idx] = make_cuDoubleComplex(start + step * idx, 0);
      }
    }
    __global__ void cuSetArange_kernel(cuFloatComplex *in, cytnx_double start, cytnx_double step,
                                       cytnx_uint64 Nelem) {
      const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      if (idx < Nelem) {
        in[idx] = make_cuFloatComplex(start + step * idx, 0);
      }
    }

    // type = 0, start < end , incremental
    // type = 1, start > end , decremental
    void cuSetArange_gpu_cd(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                            const cytnx_double &end, const cytnx_double &step,
                            const cytnx_uint64 &Nelem) {
      cuDoubleComplex *ptr = (cuDoubleComplex *)in->data();
      cytnx_uint64 NBlocks = Nelem / 512;

      if (Nelem % 512) NBlocks += 1;
      cuSetArange_kernel<<<NBlocks, 512>>>(ptr, start, step, Nelem);
    }
    void cuSetArange_gpu_cf(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                            const cytnx_double &end, const cytnx_double &step,
                            const cytnx_uint64 &Nelem) {
      cuFloatComplex *ptr = (cuFloatComplex *)in->data();
      cytnx_uint64 NBlocks = Nelem / 512;

      if (Nelem % 512) NBlocks += 1;
      cuSetArange_kernel<<<NBlocks, 512>>>(ptr, start, step, Nelem);
    }
    void cuSetArange_gpu_d(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem) {
      cytnx_double *ptr = (cytnx_double *)in->data();
      cytnx_uint64 NBlocks = Nelem / 512;

      if (Nelem % 512) NBlocks += 1;
      cuSetArange_kernel<<<NBlocks, 512>>>(ptr, start, step, Nelem);
    }
    void cuSetArange_gpu_f(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem) {
      cytnx_float *ptr = (cytnx_float *)in->data();
      cytnx_uint64 NBlocks = Nelem / 512;

      if (Nelem % 512) NBlocks += 1;
      cuSetArange_kernel<<<NBlocks, 512>>>(ptr, start, step, Nelem);
    }
    void cuSetArange_gpu_i64(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                             const cytnx_double &end, const cytnx_double &step,
                             const cytnx_uint64 &Nelem) {
      cytnx_int64 *ptr = (cytnx_int64 *)in->data();
      cytnx_uint64 NBlocks = Nelem / 512;

      if (Nelem % 512) NBlocks += 1;
      cuSetArange_kernel<<<NBlocks, 512>>>(ptr, start, step, Nelem);
    }
    void cuSetArange_gpu_u64(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                             const cytnx_double &end, const cytnx_double &step,
                             const cytnx_uint64 &Nelem) {
      cytnx_uint64 *ptr = (cytnx_uint64 *)in->data();
      cytnx_uint64 NBlocks = Nelem / 512;

      if (Nelem % 512) NBlocks += 1;
      cuSetArange_kernel<<<NBlocks, 512>>>(ptr, start, step, Nelem);
    }
    void cuSetArange_gpu_i32(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                             const cytnx_double &end, const cytnx_double &step,
                             const cytnx_uint64 &Nelem) {
      cytnx_int32 *ptr = (cytnx_int32 *)in->data();
      cytnx_uint64 NBlocks = Nelem / 512;

      if (Nelem % 512) NBlocks += 1;
      cuSetArange_kernel<<<NBlocks, 512>>>(ptr, start, step, Nelem);
    }
    void cuSetArange_gpu_u32(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                             const cytnx_double &end, const cytnx_double &step,
                             const cytnx_uint64 &Nelem) {
      cytnx_uint32 *ptr = (cytnx_uint32 *)in->data();
      cytnx_uint64 NBlocks = Nelem / 512;

      if (Nelem % 512) NBlocks += 1;
      cuSetArange_kernel<<<NBlocks, 512>>>(ptr, start, step, Nelem);
    }
    void cuSetArange_gpu_i16(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                             const cytnx_double &end, const cytnx_double &step,
                             const cytnx_uint64 &Nelem) {
      cytnx_int16 *ptr = (cytnx_int16 *)in->data();
      cytnx_uint64 NBlocks = Nelem / 512;

      if (Nelem % 512) NBlocks += 1;
      cuSetArange_kernel<<<NBlocks, 512>>>(ptr, start, step, Nelem);
    }
    void cuSetArange_gpu_u16(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                             const cytnx_double &end, const cytnx_double &step,
                             const cytnx_uint64 &Nelem) {
      cytnx_uint16 *ptr = (cytnx_uint16 *)in->data();
      cytnx_uint64 NBlocks = Nelem / 512;

      if (Nelem % 512) NBlocks += 1;
      cuSetArange_kernel<<<NBlocks, 512>>>(ptr, start, step, Nelem);
    }
    void cuSetArange_gpu_b(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem) {
      cytnx_bool *ptr = (cytnx_bool *)in->data();
      cytnx_uint64 NBlocks = Nelem / 512;

      if (Nelem % 512) NBlocks += 1;
      cuSetArange_kernel<<<NBlocks, 512>>>(ptr, start, step, Nelem);
    }

  }  // namespace utils_internal
}  // namespace cytnx
