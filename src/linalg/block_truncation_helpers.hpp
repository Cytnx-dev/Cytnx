#pragma once
#include "Tensor.hpp"
#include "UniTensor.hpp"

namespace cytnx {
  namespace linalg {

    // Returns a UniTensor holding the discarded singular values for the return_err path of
    // block truncated SVD functions (Svd_truncate, Gesvd_truncate).
    //
    // Sall      : ascending-sorted tensor of singular values that were candidates for dropping
    //             (indices 0..smidx-1 were dropped, smidx..end were kept).
    // smidx     : number of dropped values (== 0 when nothing was dropped).
    // return_err: 1 → return largest dropped value; >1 → return all dropped values descending.
    //
    // When smidx == 0 (no truncation), returns a one-element zero tensor regardless of return_err.
    inline UniTensor BuildBlockDiscardedSingularValues(const Tensor &Sall,
                                                       const cytnx_uint64 smidx,
                                                       const unsigned int return_err) {
      Tensor terr({1}, Sall.dtype());
      terr.storage().at(0) = 0;
      if (smidx == 0) {
        return UniTensor(terr);
      }
      if (return_err == 1) {
        terr.storage().at(0) = Sall.storage()(smidx - 1);
        return UniTensor(terr);
      }

      terr = Tensor({smidx}, Sall.dtype());
      for (cytnx_uint64 i = 0; i < smidx; i++) {
        terr.storage().at(i) = Sall.storage()(smidx - 1 - i);
      }
      return UniTensor(terr);
    }

  }  // namespace linalg
}  // namespace cytnx
