#include "cuGetElems_gpu.hpp"

namespace cytnx {
  namespace utils_internal {

    template <class T>
    __global__ void cuGetElems_kernel(T* d_out, T* d_in, cytnx_uint64* offj, cytnx_uint64* new_offj,
                                      cytnx_uint64* locators, cytnx_uint64* picksize,
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
        d_out[blockIdx.x * blockDim.x + threadIdx.x] = d_in[Loc];
      }
    }

    void cuGetElems_gpu_cd(void* out, void* in, const std::vector<cytnx_uint64>& offj,
                           const std::vector<cytnx_uint64>& new_offj,
                           const std::vector<std::vector<cytnx_uint64>>& locators,
                           const cytnx_uint64& TotalElem) {
      // Start copy elem:
      cuDoubleComplex* elem_ptr_ = static_cast<cuDoubleComplex*>(in);
      cuDoubleComplex* new_elem_ptr_ = static_cast<cuDoubleComplex*>(out);

      // create on device:
      cytnx_uint64* d_offj;
      checkCudaErrors(cudaMalloc((void**)&d_offj, sizeof(cytnx_uint64) * offj.size()));
      checkCudaErrors(
        cudaMemcpy(d_offj, &offj[0], sizeof(cytnx_uint64) * offj.size(), cudaMemcpyHostToDevice));

      cytnx_uint64* d_new_offj;
      checkCudaErrors(cudaMalloc((void**)&d_new_offj, sizeof(cytnx_uint64) * new_offj.size()));
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

      cytnx_uint64* d_locators;
      checkCudaErrors(
        cudaMalloc((void**)&d_locators, sizeof(cytnx_uint64) * composit_locators.size()));
      checkCudaErrors(cudaMemcpy(d_locators, &composit_locators[0],
                                 sizeof(cytnx_uint64) * composit_locators.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64* d_picksize;
      checkCudaErrors(cudaMalloc((void**)&d_picksize, sizeof(cytnx_uint64) * picksize.size()));
      checkCudaErrors(cudaMemcpy(d_picksize, &picksize[0], sizeof(cytnx_uint64) * picksize.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64 NBlocks = TotalElem / 256;
      if (TotalElem % 256) NBlocks += 1;
      cuGetElems_kernel<<<NBlocks, 256>>>(new_elem_ptr_, elem_ptr_, d_offj, d_new_offj, d_locators,
                                          d_picksize, offj.size(), TotalElem);

      cudaFree(d_offj);
      cudaFree(d_new_offj);
      cudaFree(d_locators);
      cudaFree(d_picksize);
    }
    void cuGetElems_gpu_cf(void* out, void* in, const std::vector<cytnx_uint64>& offj,
                           const std::vector<cytnx_uint64>& new_offj,
                           const std::vector<std::vector<cytnx_uint64>>& locators,
                           const cytnx_uint64& TotalElem) {
      // Start copy elem:
      cuFloatComplex* elem_ptr_ = static_cast<cuFloatComplex*>(in);
      cuFloatComplex* new_elem_ptr_ = static_cast<cuFloatComplex*>(out);

      // create on device:
      cytnx_uint64* d_offj;
      checkCudaErrors(cudaMalloc((void**)&d_offj, sizeof(cytnx_uint64) * offj.size()));
      checkCudaErrors(
        cudaMemcpy(d_offj, &offj[0], sizeof(cytnx_uint64) * offj.size(), cudaMemcpyHostToDevice));

      cytnx_uint64* d_new_offj;
      checkCudaErrors(cudaMalloc((void**)&d_new_offj, sizeof(cytnx_uint64) * new_offj.size()));
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

      cytnx_uint64* d_locators;
      checkCudaErrors(
        cudaMalloc((void**)&d_locators, sizeof(cytnx_uint64) * composit_locators.size()));
      checkCudaErrors(cudaMemcpy(d_locators, &composit_locators[0],
                                 sizeof(cytnx_uint64) * composit_locators.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64* d_picksize;
      checkCudaErrors(cudaMalloc((void**)&d_picksize, sizeof(cytnx_uint64) * picksize.size()));
      checkCudaErrors(cudaMemcpy(d_picksize, &picksize[0], sizeof(cytnx_uint64) * picksize.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64 NBlocks = TotalElem / 256;
      if (TotalElem % 256) NBlocks += 1;
      cuGetElems_kernel<<<NBlocks, 256>>>(new_elem_ptr_, elem_ptr_, d_offj, d_new_offj, d_locators,
                                          d_picksize, offj.size(), TotalElem);

      cudaFree(d_offj);
      cudaFree(d_new_offj);
      cudaFree(d_locators);
      cudaFree(d_picksize);
    }
    void cuGetElems_gpu_d(void* out, void* in, const std::vector<cytnx_uint64>& offj,
                          const std::vector<cytnx_uint64>& new_offj,
                          const std::vector<std::vector<cytnx_uint64>>& locators,
                          const cytnx_uint64& TotalElem) {
      // Start copy elem:
      cytnx_double* elem_ptr_ = static_cast<cytnx_double*>(in);
      cytnx_double* new_elem_ptr_ = static_cast<cytnx_double*>(out);

      // create on device:
      cytnx_uint64* d_offj;
      checkCudaErrors(cudaMalloc((void**)&d_offj, sizeof(cytnx_uint64) * offj.size()));
      checkCudaErrors(
        cudaMemcpy(d_offj, &offj[0], sizeof(cytnx_uint64) * offj.size(), cudaMemcpyHostToDevice));

      cytnx_uint64* d_new_offj;
      checkCudaErrors(cudaMalloc((void**)&d_new_offj, sizeof(cytnx_uint64) * new_offj.size()));
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

      cytnx_uint64* d_locators;
      checkCudaErrors(
        cudaMalloc((void**)&d_locators, sizeof(cytnx_uint64) * composit_locators.size()));
      checkCudaErrors(cudaMemcpy(d_locators, &composit_locators[0],
                                 sizeof(cytnx_uint64) * composit_locators.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64* d_picksize;
      checkCudaErrors(cudaMalloc((void**)&d_picksize, sizeof(cytnx_uint64) * picksize.size()));
      checkCudaErrors(cudaMemcpy(d_picksize, &picksize[0], sizeof(cytnx_uint64) * picksize.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64 NBlocks = TotalElem / 256;
      if (TotalElem % 256) NBlocks += 1;
      cuGetElems_kernel<<<NBlocks, 256>>>(new_elem_ptr_, elem_ptr_, d_offj, d_new_offj, d_locators,
                                          d_picksize, offj.size(), TotalElem);

      cudaFree(d_offj);
      cudaFree(d_new_offj);
      cudaFree(d_locators);
      cudaFree(d_picksize);
    }
    void cuGetElems_gpu_f(void* out, void* in, const std::vector<cytnx_uint64>& offj,
                          const std::vector<cytnx_uint64>& new_offj,
                          const std::vector<std::vector<cytnx_uint64>>& locators,
                          const cytnx_uint64& TotalElem) {
      // Start copy elem:
      cytnx_float* elem_ptr_ = static_cast<cytnx_float*>(in);
      cytnx_float* new_elem_ptr_ = static_cast<cytnx_float*>(out);

      // create on device:
      cytnx_uint64* d_offj;
      checkCudaErrors(cudaMalloc((void**)&d_offj, sizeof(cytnx_uint64) * offj.size()));
      checkCudaErrors(
        cudaMemcpy(d_offj, &offj[0], sizeof(cytnx_uint64) * offj.size(), cudaMemcpyHostToDevice));

      cytnx_uint64* d_new_offj;
      checkCudaErrors(cudaMalloc((void**)&d_new_offj, sizeof(cytnx_uint64) * new_offj.size()));
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

      cytnx_uint64* d_locators;
      checkCudaErrors(
        cudaMalloc((void**)&d_locators, sizeof(cytnx_uint64) * composit_locators.size()));
      checkCudaErrors(cudaMemcpy(d_locators, &composit_locators[0],
                                 sizeof(cytnx_uint64) * composit_locators.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64* d_picksize;
      checkCudaErrors(cudaMalloc((void**)&d_picksize, sizeof(cytnx_uint64) * picksize.size()));
      checkCudaErrors(cudaMemcpy(d_picksize, &picksize[0], sizeof(cytnx_uint64) * picksize.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64 NBlocks = TotalElem / 256;
      if (TotalElem % 256) NBlocks += 1;
      cuGetElems_kernel<<<NBlocks, 256>>>(new_elem_ptr_, elem_ptr_, d_offj, d_new_offj, d_locators,
                                          d_picksize, offj.size(), TotalElem);

      cudaFree(d_offj);
      cudaFree(d_new_offj);
      cudaFree(d_locators);
      cudaFree(d_picksize);
    }
    void cuGetElems_gpu_i64(void* out, void* in, const std::vector<cytnx_uint64>& offj,
                            const std::vector<cytnx_uint64>& new_offj,
                            const std::vector<std::vector<cytnx_uint64>>& locators,
                            const cytnx_uint64& TotalElem) {
      // Start copy elem:
      cytnx_int64* elem_ptr_ = static_cast<cytnx_int64*>(in);
      cytnx_int64* new_elem_ptr_ = static_cast<cytnx_int64*>(out);

      // create on device:
      cytnx_uint64* d_offj;
      checkCudaErrors(cudaMalloc((void**)&d_offj, sizeof(cytnx_uint64) * offj.size()));
      checkCudaErrors(
        cudaMemcpy(d_offj, &offj[0], sizeof(cytnx_uint64) * offj.size(), cudaMemcpyHostToDevice));

      cytnx_uint64* d_new_offj;
      checkCudaErrors(cudaMalloc((void**)&d_new_offj, sizeof(cytnx_uint64) * new_offj.size()));
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

      cytnx_uint64* d_locators;
      checkCudaErrors(
        cudaMalloc((void**)&d_locators, sizeof(cytnx_uint64) * composit_locators.size()));
      checkCudaErrors(cudaMemcpy(d_locators, &composit_locators[0],
                                 sizeof(cytnx_uint64) * composit_locators.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64* d_picksize;
      checkCudaErrors(cudaMalloc((void**)&d_picksize, sizeof(cytnx_uint64) * picksize.size()));
      checkCudaErrors(cudaMemcpy(d_picksize, &picksize[0], sizeof(cytnx_uint64) * picksize.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64 NBlocks = TotalElem / 256;
      if (TotalElem % 256) NBlocks += 1;
      cuGetElems_kernel<<<NBlocks, 256>>>(new_elem_ptr_, elem_ptr_, d_offj, d_new_offj, d_locators,
                                          d_picksize, offj.size(), TotalElem);

      cudaFree(d_offj);
      cudaFree(d_new_offj);
      cudaFree(d_locators);
      cudaFree(d_picksize);
    }
    void cuGetElems_gpu_u64(void* out, void* in, const std::vector<cytnx_uint64>& offj,
                            const std::vector<cytnx_uint64>& new_offj,
                            const std::vector<std::vector<cytnx_uint64>>& locators,
                            const cytnx_uint64& TotalElem) {
      // Start copy elem:
      cytnx_uint64* elem_ptr_ = static_cast<cytnx_uint64*>(in);
      cytnx_uint64* new_elem_ptr_ = static_cast<cytnx_uint64*>(out);

      // create on device:
      cytnx_uint64* d_offj;
      checkCudaErrors(cudaMalloc((void**)&d_offj, sizeof(cytnx_uint64) * offj.size()));
      checkCudaErrors(
        cudaMemcpy(d_offj, &offj[0], sizeof(cytnx_uint64) * offj.size(), cudaMemcpyHostToDevice));

      cytnx_uint64* d_new_offj;
      checkCudaErrors(cudaMalloc((void**)&d_new_offj, sizeof(cytnx_uint64) * new_offj.size()));
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

      cytnx_uint64* d_locators;
      checkCudaErrors(
        cudaMalloc((void**)&d_locators, sizeof(cytnx_uint64) * composit_locators.size()));
      checkCudaErrors(cudaMemcpy(d_locators, &composit_locators[0],
                                 sizeof(cytnx_uint64) * composit_locators.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64* d_picksize;
      checkCudaErrors(cudaMalloc((void**)&d_picksize, sizeof(cytnx_uint64) * picksize.size()));
      checkCudaErrors(cudaMemcpy(d_picksize, &picksize[0], sizeof(cytnx_uint64) * picksize.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64 NBlocks = TotalElem / 256;
      if (TotalElem % 256) NBlocks += 1;
      cuGetElems_kernel<<<NBlocks, 256>>>(new_elem_ptr_, elem_ptr_, d_offj, d_new_offj, d_locators,
                                          d_picksize, offj.size(), TotalElem);

      cudaFree(d_offj);
      cudaFree(d_new_offj);
      cudaFree(d_locators);
      cudaFree(d_picksize);
    }
    void cuGetElems_gpu_i32(void* out, void* in, const std::vector<cytnx_uint64>& offj,
                            const std::vector<cytnx_uint64>& new_offj,
                            const std::vector<std::vector<cytnx_uint64>>& locators,
                            const cytnx_uint64& TotalElem) {
      // Start copy elem:
      cytnx_int32* elem_ptr_ = static_cast<cytnx_int32*>(in);
      cytnx_int32* new_elem_ptr_ = static_cast<cytnx_int32*>(out);

      // create on device:
      cytnx_uint64* d_offj;
      checkCudaErrors(cudaMalloc((void**)&d_offj, sizeof(cytnx_uint64) * offj.size()));
      checkCudaErrors(
        cudaMemcpy(d_offj, &offj[0], sizeof(cytnx_uint64) * offj.size(), cudaMemcpyHostToDevice));

      cytnx_uint64* d_new_offj;
      checkCudaErrors(cudaMalloc((void**)&d_new_offj, sizeof(cytnx_uint64) * new_offj.size()));
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

      cytnx_uint64* d_locators;
      checkCudaErrors(
        cudaMalloc((void**)&d_locators, sizeof(cytnx_uint64) * composit_locators.size()));
      checkCudaErrors(cudaMemcpy(d_locators, &composit_locators[0],
                                 sizeof(cytnx_uint64) * composit_locators.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64* d_picksize;
      checkCudaErrors(cudaMalloc((void**)&d_picksize, sizeof(cytnx_uint64) * picksize.size()));
      checkCudaErrors(cudaMemcpy(d_picksize, &picksize[0], sizeof(cytnx_uint64) * picksize.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64 NBlocks = TotalElem / 256;
      if (TotalElem % 256) NBlocks += 1;
      cuGetElems_kernel<<<NBlocks, 256>>>(new_elem_ptr_, elem_ptr_, d_offj, d_new_offj, d_locators,
                                          d_picksize, offj.size(), TotalElem);

      cudaFree(d_offj);
      cudaFree(d_new_offj);
      cudaFree(d_locators);
      cudaFree(d_picksize);
    }
    void cuGetElems_gpu_u32(void* out, void* in, const std::vector<cytnx_uint64>& offj,
                            const std::vector<cytnx_uint64>& new_offj,
                            const std::vector<std::vector<cytnx_uint64>>& locators,
                            const cytnx_uint64& TotalElem) {
      // Start copy elem:
      cytnx_uint32* elem_ptr_ = static_cast<cytnx_uint32*>(in);
      cytnx_uint32* new_elem_ptr_ = static_cast<cytnx_uint32*>(out);

      // create on device:
      cytnx_uint64* d_offj;
      checkCudaErrors(cudaMalloc((void**)&d_offj, sizeof(cytnx_uint64) * offj.size()));
      checkCudaErrors(
        cudaMemcpy(d_offj, &offj[0], sizeof(cytnx_uint64) * offj.size(), cudaMemcpyHostToDevice));

      cytnx_uint64* d_new_offj;
      checkCudaErrors(cudaMalloc((void**)&d_new_offj, sizeof(cytnx_uint64) * new_offj.size()));
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

      cytnx_uint64* d_locators;
      checkCudaErrors(
        cudaMalloc((void**)&d_locators, sizeof(cytnx_uint64) * composit_locators.size()));
      checkCudaErrors(cudaMemcpy(d_locators, &composit_locators[0],
                                 sizeof(cytnx_uint64) * composit_locators.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64* d_picksize;
      checkCudaErrors(cudaMalloc((void**)&d_picksize, sizeof(cytnx_uint64) * picksize.size()));
      checkCudaErrors(cudaMemcpy(d_picksize, &picksize[0], sizeof(cytnx_uint64) * picksize.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64 NBlocks = TotalElem / 256;
      if (TotalElem % 256) NBlocks += 1;
      cuGetElems_kernel<<<NBlocks, 256>>>(new_elem_ptr_, elem_ptr_, d_offj, d_new_offj, d_locators,
                                          d_picksize, offj.size(), TotalElem);

      cudaFree(d_offj);
      cudaFree(d_new_offj);
      cudaFree(d_locators);
      cudaFree(d_picksize);
    }
    void cuGetElems_gpu_i16(void* out, void* in, const std::vector<cytnx_uint64>& offj,
                            const std::vector<cytnx_uint64>& new_offj,
                            const std::vector<std::vector<cytnx_uint64>>& locators,
                            const cytnx_uint64& TotalElem) {
      // Start copy elem:
      cytnx_int16* elem_ptr_ = static_cast<cytnx_int16*>(in);
      cytnx_int16* new_elem_ptr_ = static_cast<cytnx_int16*>(out);

      // create on device:
      cytnx_uint64* d_offj;
      checkCudaErrors(cudaMalloc((void**)&d_offj, sizeof(cytnx_uint64) * offj.size()));
      checkCudaErrors(
        cudaMemcpy(d_offj, &offj[0], sizeof(cytnx_uint64) * offj.size(), cudaMemcpyHostToDevice));

      cytnx_uint64* d_new_offj;
      checkCudaErrors(cudaMalloc((void**)&d_new_offj, sizeof(cytnx_uint64) * new_offj.size()));
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

      cytnx_uint64* d_locators;
      checkCudaErrors(
        cudaMalloc((void**)&d_locators, sizeof(cytnx_uint64) * composit_locators.size()));
      checkCudaErrors(cudaMemcpy(d_locators, &composit_locators[0],
                                 sizeof(cytnx_uint64) * composit_locators.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64* d_picksize;
      checkCudaErrors(cudaMalloc((void**)&d_picksize, sizeof(cytnx_uint64) * picksize.size()));
      checkCudaErrors(cudaMemcpy(d_picksize, &picksize[0], sizeof(cytnx_uint64) * picksize.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64 NBlocks = TotalElem / 256;
      if (TotalElem % 256) NBlocks += 1;
      cuGetElems_kernel<<<NBlocks, 256>>>(new_elem_ptr_, elem_ptr_, d_offj, d_new_offj, d_locators,
                                          d_picksize, offj.size(), TotalElem);

      cudaFree(d_offj);
      cudaFree(d_new_offj);
      cudaFree(d_locators);
      cudaFree(d_picksize);
    }
    void cuGetElems_gpu_u16(void* out, void* in, const std::vector<cytnx_uint64>& offj,
                            const std::vector<cytnx_uint64>& new_offj,
                            const std::vector<std::vector<cytnx_uint64>>& locators,
                            const cytnx_uint64& TotalElem) {
      // Start copy elem:
      cytnx_uint16* elem_ptr_ = static_cast<cytnx_uint16*>(in);
      cytnx_uint16* new_elem_ptr_ = static_cast<cytnx_uint16*>(out);

      // create on device:
      cytnx_uint64* d_offj;
      checkCudaErrors(cudaMalloc((void**)&d_offj, sizeof(cytnx_uint64) * offj.size()));
      checkCudaErrors(
        cudaMemcpy(d_offj, &offj[0], sizeof(cytnx_uint64) * offj.size(), cudaMemcpyHostToDevice));

      cytnx_uint64* d_new_offj;
      checkCudaErrors(cudaMalloc((void**)&d_new_offj, sizeof(cytnx_uint64) * new_offj.size()));
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

      cytnx_uint64* d_locators;
      checkCudaErrors(
        cudaMalloc((void**)&d_locators, sizeof(cytnx_uint64) * composit_locators.size()));
      checkCudaErrors(cudaMemcpy(d_locators, &composit_locators[0],
                                 sizeof(cytnx_uint64) * composit_locators.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64* d_picksize;
      checkCudaErrors(cudaMalloc((void**)&d_picksize, sizeof(cytnx_uint64) * picksize.size()));
      checkCudaErrors(cudaMemcpy(d_picksize, &picksize[0], sizeof(cytnx_uint64) * picksize.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64 NBlocks = TotalElem / 256;
      if (TotalElem % 256) NBlocks += 1;
      cuGetElems_kernel<<<NBlocks, 256>>>(new_elem_ptr_, elem_ptr_, d_offj, d_new_offj, d_locators,
                                          d_picksize, offj.size(), TotalElem);

      cudaFree(d_offj);
      cudaFree(d_new_offj);
      cudaFree(d_locators);
      cudaFree(d_picksize);
    }
    void cuGetElems_gpu_b(void* out, void* in, const std::vector<cytnx_uint64>& offj,
                          const std::vector<cytnx_uint64>& new_offj,
                          const std::vector<std::vector<cytnx_uint64>>& locators,
                          const cytnx_uint64& TotalElem) {
      // Start copy elem:
      cytnx_bool* elem_ptr_ = static_cast<cytnx_bool*>(in);
      cytnx_bool* new_elem_ptr_ = static_cast<cytnx_bool*>(out);

      // create on device:
      cytnx_uint64* d_offj;
      checkCudaErrors(cudaMalloc((void**)&d_offj, sizeof(cytnx_uint64) * offj.size()));
      checkCudaErrors(
        cudaMemcpy(d_offj, &offj[0], sizeof(cytnx_uint64) * offj.size(), cudaMemcpyHostToDevice));

      cytnx_uint64* d_new_offj;
      checkCudaErrors(cudaMalloc((void**)&d_new_offj, sizeof(cytnx_uint64) * new_offj.size()));
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

      cytnx_uint64* d_locators;
      checkCudaErrors(
        cudaMalloc((void**)&d_locators, sizeof(cytnx_uint64) * composit_locators.size()));
      checkCudaErrors(cudaMemcpy(d_locators, &composit_locators[0],
                                 sizeof(cytnx_uint64) * composit_locators.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64* d_picksize;
      checkCudaErrors(cudaMalloc((void**)&d_picksize, sizeof(cytnx_uint64) * picksize.size()));
      checkCudaErrors(cudaMemcpy(d_picksize, &picksize[0], sizeof(cytnx_uint64) * picksize.size(),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64 NBlocks = TotalElem / 256;
      if (TotalElem % 256) NBlocks += 1;
      cuGetElems_kernel<<<NBlocks, 256>>>(new_elem_ptr_, elem_ptr_, d_offj, d_new_offj, d_locators,
                                          d_picksize, offj.size(), TotalElem);

      cudaFree(d_offj);
      cudaFree(d_new_offj);
      cudaFree(d_locators);
      cudaFree(d_picksize);
    }

  }  // namespace utils_internal
}  // namespace cytnx
