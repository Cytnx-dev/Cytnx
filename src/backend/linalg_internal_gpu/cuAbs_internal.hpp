#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUABS_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUABS_INTERNAL_H_

#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    /// cuAbs: typed GPU dispatch (#1003). `out` must be pre-allocated with the Abs output dtype
    /// (Abs(complex) -> real; every other dtype -> itself).
    void cuAbs_dispatch(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &in, const cytnx_uint64 &Nelem);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUABS_INTERNAL_H_
