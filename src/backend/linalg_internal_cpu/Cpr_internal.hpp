#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_CPR_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_CPR_INTERNAL_H_

#include <assert.h>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"
#include "utils/utils.hpp"

namespace cytnx {
  namespace linalg_internal {
    // Cpr's output dtype is always cytnx_bool regardless of TL/TR -- unlike
    // Add/Sub/Mul/Div/Mod there is no output-type template parameter here,
    // since there is no choice to make (#941's dispatch-boundary-computes-TO
    // rule collapses to a constant for comparison operators).
    template <typename TL, typename TR>
    inline void CprInternalImpl(const boost::intrusive_ptr<StorageImplementation<cytnx_bool>> &out,
                                const boost::intrusive_ptr<StorageImplementation<TL>> &lhs,
                                const boost::intrusive_ptr<StorageImplementation<TR>> &rhs,
                                const cytnx_uint64 &len, const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      using TPromote = cytnx::Type_class::type_promote_t<TL, TR>;
      cytnx_bool *_out = out->data();
      const TL *_Lin = lhs->data();
      const TR *_Rin = rhs->data();

      if (lhs->size() == 1) {
        for (cytnx::cytnx_uint64 i = 0; i < len; i++) {
          _out[i] = (static_cast<TPromote>(_Lin[0]) == static_cast<TPromote>(_Rin[i]));
        }
      } else if (rhs->size() == 1) {
        for (cytnx::cytnx_uint64 i = 0; i < len; i++) {
          _out[i] = (static_cast<TPromote>(_Lin[i]) == static_cast<TPromote>(_Rin[0]));
        }
      } else {
        if (shape.size() == 0) {
          // Contiguous case
          for (cytnx::cytnx_uint64 i = 0; i < len; i++) {
            _out[i] = (static_cast<TPromote>(_Lin[i]) == static_cast<TPromote>(_Rin[i]));
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
            _out[i] = (static_cast<TPromote>(_Lin[idx_L]) == static_cast<TPromote>(_Rin[idx_R]));
          }
        }
      }
    }

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_CPR_INTERNAL_H_
