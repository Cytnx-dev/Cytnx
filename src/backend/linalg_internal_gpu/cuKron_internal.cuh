#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUKRON_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUKRON_INTERNAL_H_

#include "backend/utils_internal_interface.hpp"
#include "utils/utils.hpp"
#include "backend/utils_internal_gpu/cuAlloc_gpu.hpp"
#include <algorithm>

#include "cytnx_error.hpp"
#include "Type.hpp"
#include "backend/lapack_wrapper.hpp"
#include <iostream>

#ifndef __CUDACC__
  #error This file requires a CUDA compiler
#endif

namespace cytnx {

  namespace linalg_internal {

    template <class TO, class TL, class TR>
    __global__ void cuKron_kernel(TO *out, const TL *Lin, const TR *Rin, cytnx_uint64 *meta_infos,
                                  int len_nsa, int offset1, int offset2, int offset3,
                                  cytnx_uint64 Nelem) {
      extern __shared__ cytnx_uint64 info[];

      // copy data into shared mem:
      if (threadIdx.x < offset3) {
        info[threadIdx.x] = meta_infos[threadIdx.x];
      }
      __syncthreads();

      // cytnx_uint64 *new_shape_acc = info;
      // cytnx_uint64 *shape1_acc = &info[len_nsa];
      // cytnx_uint64 *shape2_acc = &info[offset1];
      // cytnx_uint64 *shape2 = &info[offset2];

      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        cytnx_uint64 tmp = blockIdx.x * blockDim.x + threadIdx.x;
        cytnx_uint64 tmp2;
        cytnx_uint64 x = 0, y = 0;
        for (cytnx_uint64 j = 0; j < len_nsa; j++) {
          tmp2 = tmp / info[j];
          tmp %= info[j];
          x += cytnx_uint64(tmp2 / info[offset2 + j]) * info[len_nsa + j];
          y += cytnx_uint64(tmp2 % info[offset2 + j]) * info[offset1 + j];
        }
        out[blockIdx.x * blockDim.x + threadIdx.x] = Lin[x] * Rin[y];
      }
    }

#define _TNinB_KRON_ 256

    template <class TO, class TL, class TR>
    void cuKron_general(TO *out, const TL *Lin, const TR *Rin,
                        const std::vector<cytnx_uint64> &shape1,
                        const std::vector<cytnx_uint64> &shape2) {
      cytnx_error_msg(shape1.size() != shape2.size(),
                      "[ERROR][Internal cuKron] T1 rank != T2 rank %s", "\n");

      std::vector<cytnx_uint64> new_shape_acc(shape1.size());
      std::vector<cytnx_uint64> shape1_acc(shape1.size());
      std::vector<cytnx_uint64> shape2_acc(shape2.size());
      new_shape_acc.back() = 1;
      shape1_acc.back() = 1;
      shape2_acc.back() = 1;
      cytnx_uint64 TotalElem = shape1[0] * shape2[0];

      for (unsigned long long i = 1; i < new_shape_acc.size(); i++) {
        new_shape_acc[new_shape_acc.size() - 1 - i] = new_shape_acc[new_shape_acc.size() - i] *
                                                      shape1[new_shape_acc.size() - i] *
                                                      shape2[new_shape_acc.size() - i];
        TotalElem *= shape1[i] * shape2[i];
        shape1_acc[shape1_acc.size() - 1 - i] =
          shape1_acc[shape1_acc.size() - i] * shape1[shape1_acc.size() - i];
        shape2_acc[shape2_acc.size() - 1 - i] =
          shape2_acc[shape2_acc.size() - i] * shape2[shape2_acc.size() - i];
      }

      int offset1 = new_shape_acc.size() + shape1_acc.size();
      int offset2 = offset1 + shape2_acc.size();
      int offset3 = offset2 + shape2.size();

      cytnx_uint64 *dshare_info = (cytnx_uint64 *)utils_internal::cuMalloc_gpu(
        (new_shape_acc.size() + shape1_acc.size() + shape2_acc.size() + shape2.size()) *
        sizeof(cytnx_uint64));
      checkCudaErrors(cudaMemcpy(dshare_info, &new_shape_acc[0],
                                 new_shape_acc.size() * sizeof(cytnx_uint64),
                                 cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy((char *)dshare_info + new_shape_acc.size() * sizeof(cytnx_uint64),
                                 shape1_acc.data(), shape1_acc.size() * sizeof(cytnx_uint64),
                                 cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy((char *)dshare_info + offset1 * sizeof(cytnx_uint64),
                                 shape2_acc.data(), shape2_acc.size() * sizeof(cytnx_uint64),
                                 cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy((char *)dshare_info + offset2 * sizeof(cytnx_uint64),
                                 shape2.data(), shape2.size() * sizeof(cytnx_uint64),
                                 cudaMemcpyHostToDevice));

      cytnx_uint64 NBlocks = TotalElem / _TNinB_KRON_;
      if (TotalElem % _TNinB_KRON_) NBlocks += 1;

      // cudaDeviceSynchronize(); // not needed, cudaMemcpy is synchronous
      cuKron_kernel<<<NBlocks, _TNinB_KRON_, offset3 * sizeof(cytnx_uint64)>>>(
        out, Lin, Rin, dshare_info, new_shape_acc.size(), offset1, offset2, offset3, TotalElem);

      cudaFree(dshare_info);
    }

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUKRON_INTERNAL_H_
