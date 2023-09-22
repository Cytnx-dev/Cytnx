#include "cuSetElems_gpu.hpp"

namespace cytnx {
  namespace utils_internal {

    // r r
    template <class T1, class T2>
    __global__ void cuSetElems_gpu_kernel(T1 *d_in, T2 *d_out, cytnx_uint64 *offj,
                                          cytnx_uint64 *new_offj, cytnx_uint64 *locators,
                                          cytnx_uint64 *picksize, cytnx_uint64 rank,
                                          cytnx_uint64 TotalElem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < TotalElem) {
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = blockIdx.x * blockDim.x + threadIdx.x;
        cytnx_uint64 offset = 0;
        for (cytnx_uint32 r = 0; r < rank; r++) {
          if (picksize[r])
            Loc += locators[offset + cytnx_uint64(tmpn / new_offj[r])] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
          offset += picksize[r];
        }
        d_out[Loc] = d_in[blockIdx.x * blockDim.x + threadIdx.x];
      }
    }
    // cd cd
    __global__ void cuSetElems_gpu_kernel(cuDoubleComplex *d_in, cuDoubleComplex *d_out,
                                          cytnx_uint64 *offj, cytnx_uint64 *new_offj,
                                          cytnx_uint64 *locators, cytnx_uint64 *picksize,
                                          cytnx_uint64 rank, cytnx_uint64 TotalElem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < TotalElem) {
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = blockIdx.x * blockDim.x + threadIdx.x;
        cytnx_uint64 offset = 0;
        for (cytnx_uint32 r = 0; r < rank; r++) {
          if (picksize[r])
            Loc += locators[offset + cytnx_uint64(tmpn / new_offj[r])] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
          offset += picksize[r];
        }
        d_out[Loc] = d_in[blockIdx.x * blockDim.x + threadIdx.x];
      }
    }
    // cd cf
    __global__ void cuSetElems_gpu_kernel(cuDoubleComplex *d_in, cuFloatComplex *d_out,
                                          cytnx_uint64 *offj, cytnx_uint64 *new_offj,
                                          cytnx_uint64 *locators, cytnx_uint64 *picksize,
                                          cytnx_uint64 rank, cytnx_uint64 TotalElem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < TotalElem) {
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = blockIdx.x * blockDim.x + threadIdx.x;
        cytnx_uint64 offset = 0;
        for (cytnx_uint32 r = 0; r < rank; r++) {
          if (picksize[r])
            Loc += locators[offset + cytnx_uint64(tmpn / new_offj[r])] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
          offset += picksize[r];
        }
        d_out[Loc] = cuComplexDoubleToFloat(d_in[blockIdx.x * blockDim.x + threadIdx.x]);
      }
    }
    // cf cd
    __global__ void cuSetElems_gpu_kernel(cuFloatComplex *d_in, cuDoubleComplex *d_out,
                                          cytnx_uint64 *offj, cytnx_uint64 *new_offj,
                                          cytnx_uint64 *locators, cytnx_uint64 *picksize,
                                          cytnx_uint64 rank, cytnx_uint64 TotalElem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < TotalElem) {
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = blockIdx.x * blockDim.x + threadIdx.x;
        cytnx_uint64 offset = 0;
        for (cytnx_uint32 r = 0; r < rank; r++) {
          if (picksize[r])
            Loc += locators[offset + cytnx_uint64(tmpn / new_offj[r])] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
          offset += picksize[r];
        }
        d_out[Loc] = cuComplexFloatToDouble(d_in[blockIdx.x * blockDim.x + threadIdx.x]);
      }
    }
    // cf cf
    __global__ void cuSetElems_gpu_kernel(cuFloatComplex *d_in, cuFloatComplex *d_out,
                                          cytnx_uint64 *offj, cytnx_uint64 *new_offj,
                                          cytnx_uint64 *locators, cytnx_uint64 *picksize,
                                          cytnx_uint64 rank, cytnx_uint64 TotalElem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < TotalElem) {
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = blockIdx.x * blockDim.x + threadIdx.x;
        cytnx_uint64 offset = 0;
        for (cytnx_uint32 r = 0; r < rank; r++) {
          if (picksize[r])
            Loc += locators[offset + cytnx_uint64(tmpn / new_offj[r])] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
          offset += picksize[r];
        }
        d_out[Loc] = d_in[blockIdx.x * blockDim.x + threadIdx.x];
      }
    }

    template <class TT>
    __global__ void cuSetElems_gpu_kernel(TT *d_in, cuDoubleComplex *d_out, cytnx_uint64 *offj,
                                          cytnx_uint64 *new_offj, cytnx_uint64 *locators,
                                          cytnx_uint64 *picksize, cytnx_uint64 rank,
                                          cytnx_uint64 TotalElem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < TotalElem) {
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = blockIdx.x * blockDim.x + threadIdx.x;
        cytnx_uint64 offset = 0;
        for (cytnx_uint32 r = 0; r < rank; r++) {
          if (picksize[r])
            Loc += locators[offset + cytnx_uint64(tmpn / new_offj[r])] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
          offset += picksize[r];
        }
        d_out[Loc].x = d_in[blockIdx.x * blockDim.x + threadIdx.x];
      }
    }

    template <class TT>
    __global__ void cuSetElems_gpu_kernel(TT *d_in, cuFloatComplex *d_out, cytnx_uint64 *offj,
                                          cytnx_uint64 *new_offj, cytnx_uint64 *locators,
                                          cytnx_uint64 *picksize, cytnx_uint64 rank,
                                          cytnx_uint64 TotalElem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < TotalElem) {
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = blockIdx.x * blockDim.x + threadIdx.x;
        cytnx_uint64 offset = 0;
        for (cytnx_uint32 r = 0; r < rank; r++) {
          if (picksize[r])
            Loc += locators[offset + cytnx_uint64(tmpn / new_offj[r])] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
          offset += picksize[r];
        }
        d_out[Loc].x = d_in[blockIdx.x * blockDim.x + threadIdx.x];
      }
    }
    //-----------------------------------------

    template <class T1, class T2>
    __global__ void cuSetElems_gpu_scal_kernel(T1 d_in, T2 *d_out, cytnx_uint64 *offj,
                                               cytnx_uint64 *new_offj, cytnx_uint64 *locators,
                                               cytnx_uint64 *picksize, cytnx_uint64 rank,
                                               cytnx_uint64 TotalElem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < TotalElem) {
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = blockIdx.x * blockDim.x + threadIdx.x;
        cytnx_uint64 offset = 0;
        for (cytnx_uint32 r = 0; r < rank; r++) {
          if (picksize[r])
            Loc += locators[offset + cytnx_uint64(tmpn / new_offj[r])] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
          offset += picksize[r];
        }
        d_out[Loc] = d_in;
      }
    }

    __global__ void cuSetElems_gpu_scal_kernel(cuDoubleComplex d_in, cuDoubleComplex *d_out,
                                               cytnx_uint64 *offj, cytnx_uint64 *new_offj,
                                               cytnx_uint64 *locators, cytnx_uint64 *picksize,
                                               cytnx_uint64 rank, cytnx_uint64 TotalElem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < TotalElem) {
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = blockIdx.x * blockDim.x + threadIdx.x;
        cytnx_uint64 offset = 0;
        for (cytnx_uint32 r = 0; r < rank; r++) {
          if (picksize[r])
            Loc += locators[offset + cytnx_uint64(tmpn / new_offj[r])] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
          offset += picksize[r];
        }
        d_out[Loc] = d_in;
      }
    }

    __global__ void cuSetElems_gpu_scal_kernel(cuDoubleComplex d_in, cuFloatComplex *d_out,
                                               cytnx_uint64 *offj, cytnx_uint64 *new_offj,
                                               cytnx_uint64 *locators, cytnx_uint64 *picksize,
                                               cytnx_uint64 rank, cytnx_uint64 TotalElem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < TotalElem) {
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = blockIdx.x * blockDim.x + threadIdx.x;
        cytnx_uint64 offset = 0;
        for (cytnx_uint32 r = 0; r < rank; r++) {
          if (picksize[r])
            Loc += locators[offset + cytnx_uint64(tmpn / new_offj[r])] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
          offset += picksize[r];
        }
        d_out[Loc] = cuComplexDoubleToFloat(d_in);
      }
    }

    __global__ void cuSetElems_gpu_scal_kernel(cuFloatComplex d_in, cuDoubleComplex *d_out,
                                               cytnx_uint64 *offj, cytnx_uint64 *new_offj,
                                               cytnx_uint64 *locators, cytnx_uint64 *picksize,
                                               cytnx_uint64 rank, cytnx_uint64 TotalElem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < TotalElem) {
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = blockIdx.x * blockDim.x + threadIdx.x;
        cytnx_uint64 offset = 0;
        for (cytnx_uint32 r = 0; r < rank; r++) {
          if (picksize[r])
            Loc += locators[offset + cytnx_uint64(tmpn / new_offj[r])] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
          offset += picksize[r];
        }
        d_out[Loc] = cuComplexFloatToDouble(d_in);
      }
    }

    __global__ void cuSetElems_gpu_scal_kernel(cuFloatComplex d_in, cuFloatComplex *d_out,
                                               cytnx_uint64 *offj, cytnx_uint64 *new_offj,
                                               cytnx_uint64 *locators, cytnx_uint64 *picksize,
                                               cytnx_uint64 rank, cytnx_uint64 TotalElem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < TotalElem) {
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = blockIdx.x * blockDim.x + threadIdx.x;
        cytnx_uint64 offset = 0;
        for (cytnx_uint32 r = 0; r < rank; r++) {
          if (picksize[r])
            Loc += locators[offset + cytnx_uint64(tmpn / new_offj[r])] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
          offset += picksize[r];
        }
        d_out[Loc] = d_in;
      }
    }

    template <class TT>
    __global__ void cuSetElems_gpu_scal_kernel(TT d_in, cuFloatComplex *d_out, cytnx_uint64 *offj,
                                               cytnx_uint64 *new_offj, cytnx_uint64 *locators,
                                               cytnx_uint64 *picksize, cytnx_uint64 rank,
                                               cytnx_uint64 TotalElem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < TotalElem) {
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = blockIdx.x * blockDim.x + threadIdx.x;
        cytnx_uint64 offset = 0;
        for (cytnx_uint32 r = 0; r < rank; r++) {
          if (picksize[r])
            Loc += locators[offset + cytnx_uint64(tmpn / new_offj[r])] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
          offset += picksize[r];
        }
        d_out[Loc].x = d_in;
      }
    }

    template <class TT>
    __global__ void cuSetElems_gpu_scal_kernel(TT d_in, cuDoubleComplex *d_out, cytnx_uint64 *offj,
                                               cytnx_uint64 *new_offj, cytnx_uint64 *locators,
                                               cytnx_uint64 *picksize, cytnx_uint64 rank,
                                               cytnx_uint64 TotalElem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < TotalElem) {
        cytnx_uint64 Loc = 0;
        cytnx_uint64 tmpn = blockIdx.x * blockDim.x + threadIdx.x;
        cytnx_uint64 offset = 0;
        for (cytnx_uint32 r = 0; r < rank; r++) {
          if (picksize[r])
            Loc += locators[offset + cytnx_uint64(tmpn / new_offj[r])] * offj[r];
          else
            Loc += cytnx_uint64(tmpn / new_offj[r]) * offj[r];
          tmpn %= new_offj[r];
          offset += picksize[r];
        }
        d_out[Loc].x = d_in;
      }
    }

    //----------------======================
    template <class Ty1, class Ty2>
    void cuSetElems_gpu_impl(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem) {
      Ty1 *new_elem_ptr_ = static_cast<Ty1 *>(in);
      Ty2 *elem_ptr_ = static_cast<Ty2 *>(out);

      // create on device:
      cytnx_uint64 *d_offj;
      checkCudaErrors(cudaMalloc((void **)&d_offj, sizeof(cytnx_uint64) * offj.size()));
      checkCudaErrors(
        cudaMemcpy(d_offj, &offj[0], sizeof(cytnx_uint64) * offj.size(), cudaMemcpyHostToDevice));

      cytnx_uint64 *d_new_offj;
      checkCudaErrors(cudaMalloc((void **)&d_new_offj, sizeof(cytnx_uint64) * new_offj.size()));
      checkCudaErrors(cudaMemcpy(d_new_offj, &new_offj[0], sizeof(cytnx_uint64) * new_offj.size(),
                                 cudaMemcpyHostToDevice));

      std::vector<cytnx_uint64> composit_locators, picksize;
      cytnx_uint64 Nte = 0;
      for (cytnx_uint32 i = 0; i < locators.size(); i++) {
        picksize.push_back(locators[i].size());
        Nte += locators[i].size();
      }
      composit_locators.resize(Nte);
      Nte = 0;
      for (cytnx_uint32 i = 0; i < locators.size(); i++) {
        memcpy(&composit_locators[Nte], &(locators[i][0]),
               sizeof(cytnx_uint64) * locators[i].size());
        Nte += locators[i].size();
      }

      cytnx_uint64 *d_locators;
      checkCudaErrors(
        cudaMalloc((void **)&d_locators, sizeof(cytnx_uint64) * composit_locators.size()));
      checkCudaErrors(cudaMemcpy(d_locators, &composit_locators[0],
                                 sizeof(cytnx_uint64) * composit_locators.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64 *d_picksize;
      checkCudaErrors(cudaMalloc((void **)&d_picksize, sizeof(cytnx_uint64) * picksize.size()));
      checkCudaErrors(cudaMemcpy(d_picksize, &picksize[0], sizeof(cytnx_uint64) * picksize.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64 NBlocks = TotalElem / 256;
      if (TotalElem % 256) NBlocks += 1;
      cuSetElems_gpu_kernel<<<NBlocks, 256>>>(new_elem_ptr_, elem_ptr_, d_offj, d_new_offj,
                                              d_locators, d_picksize, offj.size(), TotalElem);

      cudaFree(d_offj);
      cudaFree(d_new_offj);
      cudaFree(d_locators);
      cudaFree(d_picksize);
    }

    template <class Ty1, class Ty2>
    void cuSetElems_gpu_scal_impl(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem) {
      Ty1 new_elem_ = *(static_cast<Ty1 *>(in));
      Ty2 *elem_ptr_ = static_cast<Ty2 *>(out);

      // create on device:
      cytnx_uint64 *d_offj;
      checkCudaErrors(cudaMalloc((void **)&d_offj, sizeof(cytnx_uint64) * offj.size()));
      checkCudaErrors(
        cudaMemcpy(d_offj, &offj[0], sizeof(cytnx_uint64) * offj.size(), cudaMemcpyHostToDevice));

      cytnx_uint64 *d_new_offj;
      checkCudaErrors(cudaMalloc((void **)&d_new_offj, sizeof(cytnx_uint64) * new_offj.size()));
      checkCudaErrors(cudaMemcpy(d_new_offj, &new_offj[0], sizeof(cytnx_uint64) * new_offj.size(),
                                 cudaMemcpyHostToDevice));

      std::vector<cytnx_uint64> composit_locators, picksize;
      cytnx_uint64 Nte = 0;
      for (cytnx_uint32 i = 0; i < locators.size(); i++) {
        picksize.push_back(locators[i].size());
        Nte += locators[i].size();
      }
      composit_locators.resize(Nte);
      Nte = 0;
      for (cytnx_uint32 i = 0; i < locators.size(); i++) {
        memcpy(&composit_locators[Nte], &(locators[i][0]),
               sizeof(cytnx_uint64) * locators[i].size());
        Nte += locators[i].size();
      }

      cytnx_uint64 *d_locators;
      checkCudaErrors(
        cudaMalloc((void **)&d_locators, sizeof(cytnx_uint64) * composit_locators.size()));
      checkCudaErrors(cudaMemcpy(d_locators, &composit_locators[0],
                                 sizeof(cytnx_uint64) * composit_locators.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64 *d_picksize;
      checkCudaErrors(cudaMalloc((void **)&d_picksize, sizeof(cytnx_uint64) * picksize.size()));
      checkCudaErrors(cudaMemcpy(d_picksize, &picksize[0], sizeof(cytnx_uint64) * picksize.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64 NBlocks = TotalElem / 256;
      if (TotalElem % 256) NBlocks += 1;
      cuSetElems_gpu_scal_kernel<<<NBlocks, 256>>>(new_elem_, elem_ptr_, d_offj, d_new_offj,
                                                   d_locators, d_picksize, offj.size(), TotalElem);

      cudaFree(d_offj);
      cudaFree(d_new_offj);
      cudaFree(d_locators);
      cudaFree(d_picksize);
    }

    //=======================
    void cuSetElems_gpu_cdtcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cuDoubleComplex, cuDoubleComplex>(in, out, offj, new_offj,
                                                                   locators, TotalElem);
      else
        cuSetElems_gpu_impl<cuDoubleComplex, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                              TotalElem);
    }
    void cuSetElems_gpu_cdtcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cuDoubleComplex, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                                  TotalElem);
      else
        cuSetElems_gpu_impl<cuDoubleComplex, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                             TotalElem);
    }

    //----
    void cuSetElems_gpu_cftcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cuFloatComplex, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                                  TotalElem);
      else
        cuSetElems_gpu_impl<cuFloatComplex, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                             TotalElem);
    }
    void cuSetElems_gpu_cftcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cuFloatComplex, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                                 TotalElem);
      else
        cuSetElems_gpu_impl<cuFloatComplex, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                            TotalElem);
    }

    //----
    void cuSetElems_gpu_dtcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_double, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                                TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_double, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                           TotalElem);
    }
    void cuSetElems_gpu_dtcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_double, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                               TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_double, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                          TotalElem);
    }
    void cuSetElems_gpu_dtd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_double, cytnx_double>(in, out, offj, new_offj, locators,
                                                             TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_double, cytnx_double>(in, out, offj, new_offj, locators,
                                                        TotalElem);
    }
    void cuSetElems_gpu_dtf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_double, cytnx_float>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_double, cytnx_float>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_dti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_double, cytnx_int64>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_double, cytnx_int64>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_dtu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_double, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                             TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_double, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                        TotalElem);
    }
    void cuSetElems_gpu_dti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_double, cytnx_int32>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_double, cytnx_int32>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_dtu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_double, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                             TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_double, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                        TotalElem);
    }
    void cuSetElems_gpu_dti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_double, cytnx_int16>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_double, cytnx_int16>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_dtu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_double, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                             TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_double, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                        TotalElem);
    }
    void cuSetElems_gpu_dtb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_double, cytnx_bool>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_double, cytnx_bool>(in, out, offj, new_offj, locators, TotalElem);
    }

    //----
    void cuSetElems_gpu_ftcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_float, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                               TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_float, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                          TotalElem);
    }
    void cuSetElems_gpu_ftcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_float, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                              TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_float, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                         TotalElem);
    }
    void cuSetElems_gpu_ftd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_float, cytnx_double>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_float, cytnx_double>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_ftf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_float, cytnx_float>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_float, cytnx_float>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_fti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_float, cytnx_int64>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_float, cytnx_int64>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_ftu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_float, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_float, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_fti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_float, cytnx_int32>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_float, cytnx_int32>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_ftu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_float, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_float, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_ftu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_float, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_float, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_fti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_float, cytnx_int16>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_float, cytnx_int16>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_ftb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_float, cytnx_bool>(in, out, offj, new_offj, locators,
                                                          TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_float, cytnx_bool>(in, out, offj, new_offj, locators, TotalElem);
    }

    //----
    void cuSetElems_gpu_i64tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int64, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                               TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int64, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                          TotalElem);
    }
    void cuSetElems_gpu_i64tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int64, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                              TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int64, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                         TotalElem);
    }
    void cuSetElems_gpu_i64td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int64, cytnx_double>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int64, cytnx_double>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_i64tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int64, cytnx_float>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int64, cytnx_float>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_i64ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int64, cytnx_int64>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int64, cytnx_int64>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_i64tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int64, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int64, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_i64ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int64, cytnx_int32>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int64, cytnx_int32>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_i64tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int64, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int64, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_i64tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int64, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int64, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_i64ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int64, cytnx_int16>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int64, cytnx_int16>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_i64tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int64, cytnx_bool>(in, out, offj, new_offj, locators,
                                                          TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int64, cytnx_bool>(in, out, offj, new_offj, locators, TotalElem);
    }

    //----
    void cuSetElems_gpu_u64tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint64, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                                TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint64, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                           TotalElem);
    }
    void cuSetElems_gpu_u64tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint64, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                               TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint64, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                          TotalElem);
    }
    void cuSetElems_gpu_u64td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint64, cytnx_double>(in, out, offj, new_offj, locators,
                                                             TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint64, cytnx_double>(in, out, offj, new_offj, locators,
                                                        TotalElem);
    }
    void cuSetElems_gpu_u64tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint64, cytnx_float>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint64, cytnx_float>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_u64ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint64, cytnx_int64>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint64, cytnx_int64>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_u64tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint64, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                             TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint64, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                        TotalElem);
    }
    void cuSetElems_gpu_u64ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint64, cytnx_int32>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint64, cytnx_int32>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_u64tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint64, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                             TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint64, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                        TotalElem);
    }
    void cuSetElems_gpu_u64tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint64, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                             TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint64, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                        TotalElem);
    }
    void cuSetElems_gpu_u64ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint64, cytnx_int32>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint64, cytnx_int32>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_u64tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint64, cytnx_bool>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint64, cytnx_bool>(in, out, offj, new_offj, locators, TotalElem);
    }

    //----
    void cuSetElems_gpu_i32tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int32, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                               TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int32, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                          TotalElem);
    }
    void cuSetElems_gpu_i32tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int32, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                              TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int32, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                         TotalElem);
    }
    void cuSetElems_gpu_i32td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int32, cytnx_double>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int32, cytnx_double>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_i32tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int32, cytnx_float>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int32, cytnx_float>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_i32ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int32, cytnx_int64>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int32, cytnx_int64>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_i32tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int32, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int32, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_i32ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int32, cytnx_int32>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int32, cytnx_int32>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_i32tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int32, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int32, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_i32tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int32, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int32, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_i32ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int32, cytnx_int16>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int32, cytnx_int16>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_i32tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int32, cytnx_bool>(in, out, offj, new_offj, locators,
                                                          TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int32, cytnx_bool>(in, out, offj, new_offj, locators, TotalElem);
    }

    //----
    void cuSetElems_gpu_u32tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint32, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                                TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint32, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                           TotalElem);
    }
    void cuSetElems_gpu_u32tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint32, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                               TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint32, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                          TotalElem);
    }
    void cuSetElems_gpu_u32td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint32, cytnx_double>(in, out, offj, new_offj, locators,
                                                             TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint32, cytnx_double>(in, out, offj, new_offj, locators,
                                                        TotalElem);
    }
    void cuSetElems_gpu_u32tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint32, cytnx_float>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint32, cytnx_float>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_u32ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint32, cytnx_int64>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint32, cytnx_int64>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_u32tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint32, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                             TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint32, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                        TotalElem);
    }
    void cuSetElems_gpu_u32ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint32, cytnx_int32>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint32, cytnx_int32>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_u32tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint32, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                             TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint32, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                        TotalElem);
    }
    void cuSetElems_gpu_u32tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint32, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                             TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint32, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                        TotalElem);
    }
    void cuSetElems_gpu_u32ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint32, cytnx_int16>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint32, cytnx_int16>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_u32tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint32, cytnx_bool>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint32, cytnx_bool>(in, out, offj, new_offj, locators, TotalElem);
    }

    //----
    void cuSetElems_gpu_i16tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int16, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                               TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int16, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                          TotalElem);
    }
    void cuSetElems_gpu_i16tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int16, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                              TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int16, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                         TotalElem);
    }
    void cuSetElems_gpu_i16td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int16, cytnx_double>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int16, cytnx_double>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_i16tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int16, cytnx_float>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int16, cytnx_float>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_i16ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int16, cytnx_int64>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int16, cytnx_int64>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_i16tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int16, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int16, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_i16ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int16, cytnx_int32>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int16, cytnx_int32>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_i16tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int16, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int16, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_i16tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int16, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int16, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_i16ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int16, cytnx_int16>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int16, cytnx_int16>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_i16tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_int16, cytnx_bool>(in, out, offj, new_offj, locators,
                                                          TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_int16, cytnx_bool>(in, out, offj, new_offj, locators, TotalElem);
    }

    //----
    void cuSetElems_gpu_u16tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint16, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                                TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint16, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                           TotalElem);
    }
    void cuSetElems_gpu_u16tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint16, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                               TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint16, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                          TotalElem);
    }
    void cuSetElems_gpu_u16td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint16, cytnx_double>(in, out, offj, new_offj, locators,
                                                             TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint16, cytnx_double>(in, out, offj, new_offj, locators,
                                                        TotalElem);
    }
    void cuSetElems_gpu_u16tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint16, cytnx_float>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint16, cytnx_float>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_u16ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint16, cytnx_int64>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint16, cytnx_int64>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_u16tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint16, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                             TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint16, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                        TotalElem);
    }
    void cuSetElems_gpu_u16ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint16, cytnx_int32>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint16, cytnx_int32>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_u16tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint16, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                             TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint16, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                        TotalElem);
    }
    void cuSetElems_gpu_u16tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint16, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                             TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint16, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                        TotalElem);
    }
    void cuSetElems_gpu_u16ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint16, cytnx_int16>(in, out, offj, new_offj, locators,
                                                            TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint16, cytnx_int16>(in, out, offj, new_offj, locators,
                                                       TotalElem);
    }
    void cuSetElems_gpu_u16tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_uint16, cytnx_bool>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_uint16, cytnx_bool>(in, out, offj, new_offj, locators, TotalElem);
    }

    //----
    void cuSetElems_gpu_btcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_bool, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                              TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_bool, cuDoubleComplex>(in, out, offj, new_offj, locators,
                                                         TotalElem);
    }
    void cuSetElems_gpu_btcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_bool, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                             TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_bool, cuFloatComplex>(in, out, offj, new_offj, locators,
                                                        TotalElem);
    }
    void cuSetElems_gpu_btd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_bool, cytnx_double>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_bool, cytnx_double>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_btf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_bool, cytnx_float>(in, out, offj, new_offj, locators,
                                                          TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_bool, cytnx_float>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_bti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_bool, cytnx_int64>(in, out, offj, new_offj, locators,
                                                          TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_bool, cytnx_int64>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_btu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_bool, cytnx_uint64>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_bool, cytnx_uint64>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_bti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_bool, cytnx_int32>(in, out, offj, new_offj, locators,
                                                          TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_bool, cytnx_int32>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_btu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_bool, cytnx_uint32>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_bool, cytnx_uint32>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_btu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_bool, cytnx_uint16>(in, out, offj, new_offj, locators,
                                                           TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_bool, cytnx_uint16>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_bti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_bool, cytnx_int16>(in, out, offj, new_offj, locators,
                                                          TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_bool, cytnx_int16>(in, out, offj, new_offj, locators, TotalElem);
    }
    void cuSetElems_gpu_btb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar) {
      if (is_scalar)
        cuSetElems_gpu_scal_impl<cytnx_bool, cytnx_bool>(in, out, offj, new_offj, locators,
                                                         TotalElem);
      else
        cuSetElems_gpu_impl<cytnx_bool, cytnx_bool>(in, out, offj, new_offj, locators, TotalElem);
    }

  }  // namespace utils_internal
}  // namespace cytnx
