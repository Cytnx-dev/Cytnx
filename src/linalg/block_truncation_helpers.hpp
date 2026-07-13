#pragma once
#include "Tensor.hpp"
#include "UniTensor.hpp"

namespace cytnx {
  namespace linalg {

    // The block truncation keeps every singular value >= Smin (the cut value selected by the
    // keepdim/err/mindim rules), so an exact degeneracy straddling the cut enlarges the kept
    // set beyond keepdim (see the degeneracy note in the Svd_truncate documentation). Given
    // the ascending-sorted Sall and the cut index smidx (with Sall[smidx] == Smin), returns
    // the number of values strictly below Smin -- the values that are actually dropped --
    // keeping the return_err output consistent with the enlarged kept set.
    inline cytnx_uint64 CountDroppedSingularValues(const Tensor &Sall, cytnx_uint64 smidx,
                                                   const Scalar &Smin) {
      while (smidx > 0 && !(Sall.storage()(smidx - 1) < Smin)) {
        --smidx;
      }
      return smidx;
    }

    // Returns a UniTensor holding the discarded singular values for the return_err path of
    // block truncated SVD functions (Svd_truncate, Gesvd_truncate).
    //
    // Sall      : ascending-sorted tensor of singular values that were candidates for dropping
    //             (indices 0..smidx-1 were dropped, smidx..end were kept).
    // smidx     : number of dropped values (== 0 when nothing was dropped).
    // return_err: 1 → return largest dropped value; >1 → return all dropped values descending.
    //
    inline UniTensor BuildNoDiscardedSingularValues(const unsigned int dtype,
                                                    const unsigned int return_err,
                                                    const int device) {
      Tensor terr = return_err == 1 ? Tensor({}, dtype, device) : Tensor({0}, dtype, device);
      if (return_err == 1) terr.storage().at(0) = 0;
      return UniTensor(terr);
    }

    // When smidx == 0 (no truncation), return_err == 1 returns a scalar zero, while
    // return_err > 1 returns an empty vector.
    inline UniTensor BuildBlockDiscardedSingularValues(const Tensor &Sall, const cytnx_uint64 smidx,
                                                       const unsigned int return_err) {
      if (smidx == 0) {
        return BuildNoDiscardedSingularValues(Sall.dtype(), return_err, Sall.device());
      }
      if (return_err == 1) {
        Tensor terr({}, Sall.dtype(), Sall.device());
        terr.storage().at(0) = Sall.storage()(smidx - 1);
        return UniTensor(terr);
      }

      Tensor terr({smidx}, Sall.dtype(), Sall.device());
      for (cytnx_uint64 i = 0; i < smidx; i++) {
        terr.storage().at(i) = Sall.storage()(smidx - 1 - i);
      }
      return UniTensor(terr);
    }

  }  // namespace linalg
}  // namespace cytnx
