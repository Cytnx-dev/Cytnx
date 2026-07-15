#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_SUB_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_SUB_INTERNAL_H_

#include <assert.h>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"
#include "utils/utils.hpp"

namespace cytnx {
  namespace linalg_internal {
    // Typed SubInternalImpl (#941): TO is part of the signature, not recovered from a
    // dtype() check at runtime -- the caller is responsible for recovering typed
    // storage (storage_cast<T> / storage_as_type_or_replace<T>) before calling in.
    template <typename TO, typename TL, typename TR>
    inline void SubInternalImpl(const boost::intrusive_ptr<StorageImplementation<TO>> &out,
                                const boost::intrusive_ptr<StorageImplementation<TL>> &lhs,
                                const boost::intrusive_ptr<StorageImplementation<TR>> &rhs,
                                const cytnx_uint64 &len, const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      TO *_out = out->data();
      const TL *_Lin = lhs->data();
      const TR *_Rin = rhs->data();

      if (lhs->size() == 1) {
        for (cytnx::cytnx_uint64 i = 0; i < len; i++) {
          _out[i] = static_cast<TO>(static_cast<TO>(_Lin[0]) - static_cast<TO>(_Rin[i]));
        }
      } else if (rhs->size() == 1) {
        for (cytnx::cytnx_uint64 i = 0; i < len; i++) {
          _out[i] = static_cast<TO>(static_cast<TO>(_Lin[i]) - static_cast<TO>(_Rin[0]));
        }
      } else {
        if (shape.size() == 0) {
          // Contiguous case
          for (cytnx::cytnx_uint64 i = 0; i < len; i++) {
            _out[i] = static_cast<TO>(static_cast<TO>(_Lin[i]) - static_cast<TO>(_Rin[i]));
          }
        } else {
          // Non-contiguous case: handle permutations
          std::vector<cytnx::cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx::cytnx_uint64> old_accu_shapeL(shape.size()),
            old_accu_shapeR(shape.size());
          cytnx::cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;

          for (cytnx::cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

          // Handle non-contiguous memory access
          for (cytnx::cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx::cytnx_uint64> tmpv = cytnx::c2cartesian(i, accu_shape);
            cytnx::cytnx_uint64 idx_L =
              cytnx::cartesian2c(cytnx::vec_map(tmpv, invmapper_L), old_accu_shapeL);
            cytnx::cytnx_uint64 idx_R =
              cytnx::cartesian2c(cytnx::vec_map(tmpv, invmapper_R), old_accu_shapeR);
            _out[i] = static_cast<TO>(static_cast<TO>(_Lin[idx_L]) - static_cast<TO>(_Rin[idx_R]));
          }
        }
      }
    }

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_SUB_INTERNAL_H_
