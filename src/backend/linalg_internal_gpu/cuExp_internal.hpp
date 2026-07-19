#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUEXP_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUEXP_INTERNAL_H_

#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    /// cuExp: typed GPU dispatch (#1003). in/out share the (floating/complex) dispatch dtype.
    void cuExp_dispatch(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &in, const cytnx_uint64 &Nelem);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUEXP_INTERNAL_H_
