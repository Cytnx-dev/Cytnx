#include "cuSplit_internal.hpp"

namespace cytnx {

  namespace algo_internal {

    void cuvSplit_internal(std::vector<char *> &targ_ptrs, const char *In_ptr,
                           const std::vector<cytnx_uint64> &dims, const cytnx_uint64 &Col_dim,
                           const cytnx_uint64 &ElemSize) {
      // targ_ptrs need to be properly allocated!
      //  targ_ptrs.size() must be the same as dims.size()

      cytnx_uint64 roffs = 0;
      for (cytnx_uint64 i = 0; i < dims.size(); i++) {
        checkCudaErrors(cudaMemcpy(targ_ptrs[i], In_ptr + roffs, ElemSize * dims[i] * Col_dim,
                                   cudaMemcpyDeviceToDevice));
        roffs += ElemSize * dims[i] * Col_dim;
      }
    }

    void cuhSplit_internal(std::vector<char *> &targ_ptrs, const char *In_ptr,
                           const std::vector<cytnx_uint64> &dims, const cytnx_uint64 &Row_dim,
                           const cytnx_uint64 &ElemSize) {
      // targ_ptrs need to be properly allocated!
      //  targ_ptrs.size() must be the same as dims.size()

      cytnx_uint64 Dtot = dims[0];
      std::vector<cytnx_uint64> offs(dims.size());
      for (int i = 1; i < offs.size(); i++) {
        offs[i] = offs[i - 1] + dims[i - 1];
      }
      Dtot = offs.back() + dims.back();

      // for each chunk:
      for (cytnx_uint64 t = 0; t < dims.size(); t++) {
        // copy row by row:
        for (cytnx_uint64 r = 0; r < Row_dim; r++) {
          checkCudaErrors(cudaMemcpy(targ_ptrs[t] + r * dims[t] * ElemSize,
                                     In_ptr + (r * Dtot + offs[t]) * ElemSize, ElemSize * dims[t],
                                     cudaMemcpyDeviceToDevice));
        }
      }
    }

  }  // namespace algo_internal

}  // namespace cytnx
