#include "Split_internal.hpp"
#include "backend/algo_internal_interface.hpp"
#include <iostream>
#include <algorithm>

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {

  namespace algo_internal {

    void vSplit_internal(std::vector<char *> &targ_ptrs, const char *In_ptr,
                         const std::vector<cytnx_uint64> &dims, const cytnx_uint64 &Col_dim,
                         const cytnx_uint64 &ElemSize) {
      // targ_ptrs need to be properly allocated!
      //  targ_ptrs.size() must be the same as dims.size()

      cytnx_uint64 roffs = 0;
      for (cytnx_uint64 i = 0; i < dims.size(); i++) {
        memcpy(targ_ptrs[i], In_ptr + roffs, ElemSize * dims[i] * Col_dim);
        roffs += ElemSize * dims[i] * Col_dim;
      }
    }

    void hSplit_internal(std::vector<char *> &targ_ptrs, const char *In_ptr,
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
          memcpy(targ_ptrs[t] + r * dims[t] * ElemSize, In_ptr + (r * Dtot + offs[t]) * ElemSize,
                 ElemSize * dims[t]);
        }
      }
    }

  }  // namespace algo_internal

}  // namespace cytnx
