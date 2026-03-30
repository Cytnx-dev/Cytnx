#ifndef CYTNX_LINALG_IARITHMETIC_VISIT_HPP_
#define CYTNX_LINALG_IARITHMETIC_VISIT_HPP_

#include <type_traits>
#include <vector>

#include "Tensor.hpp"
#include "utils/utils.hpp"

namespace cytnx {
  namespace linalg {
    namespace detail {

      template <char op_code, typename TLin, typename TRin>
      inline void ApplyInplaceArithmeticOp(TLin &lhs, const TRin &rhs) {
        if constexpr (!cytnx::is_complex_v<TLin> && cytnx::is_complex_v<TRin>) {
          if constexpr (op_code == 0) {
            cytnx_error_msg(true, "[ERROR][iadd] Cannot perform real+=complex%s", "\n");
          } else if constexpr (op_code == 1) {
            cytnx_error_msg(true, "[ERROR][imul] Cannot perform real+=complex%s", "\n");
          } else if constexpr (op_code == 2) {
            cytnx_error_msg(true, "[ERROR][isub] Cannot perform real+=complex%s", "\n");
          } else if constexpr (op_code == 3) {
            cytnx_error_msg(true, "[ERROR][idiv] Cannot perform real+=complex%s", "\n");
          }
        } else {
          if constexpr (op_code == 0) {
            lhs += rhs;
          } else if constexpr (op_code == 1) {
            lhs *= rhs;
          } else if constexpr (op_code == 2) {
            lhs -= rhs;
          } else if constexpr (op_code == 3) {
            lhs /= rhs;
          }
        }
      }

      template <char op_code, typename TLin, typename TRin>
      inline void ApplyInplaceArithmeticKernel(
        TLin *lhs, const TRin *rhs, const cytnx_uint64 &len, const bool &rhs_is_scalar,
        const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L,
        const std::vector<cytnx_uint64> &invmapper_R) {
        if (rhs_is_scalar) {
          for (cytnx_uint64 i = 0; i < len; i++) {
            ApplyInplaceArithmeticOp<op_code>(lhs[i], rhs[0]);
          }
          return;
        }

        if (shape.empty()) {
          for (cytnx_uint64 i = 0; i < len; i++) {
            ApplyInplaceArithmeticOp<op_code>(lhs[i], rhs[i]);
          }
          return;
        }

        std::vector<cytnx_uint64> accu_shape(shape.size());
        std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
        cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
        for (cytnx_uint64 i = 0; i < shape.size(); i++) {
          accu_shape[shape.size() - 1 - i] = tmp1;
          tmp1 *= shape[shape.size() - 1 - i];

          old_accu_shapeL[shape.size() - 1 - i] = tmp2;
          tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

          old_accu_shapeR[shape.size() - 1 - i] = tmp3;
          tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
        }

        for (cytnx_uint64 i = 0; i < len; i++) {
          std::vector<cytnx_uint64> tmpv = cytnx::c2cartesian(i, accu_shape);
          cytnx_uint64 idx_L =
            cytnx::cartesian2c(cytnx::vec_map(tmpv, invmapper_L), old_accu_shapeL);
          cytnx_uint64 idx_R =
            cytnx::cartesian2c(cytnx::vec_map(tmpv, invmapper_R), old_accu_shapeR);
          ApplyInplaceArithmeticOp<op_code>(lhs[idx_L], rhs[idx_R]);
        }
      }

      template <char op_code>
      inline void DispatchInplaceArithmeticCPU(Tensor &Lt, const Tensor &Rt,
                                               const std::vector<cytnx_uint64> &shape,
                                               const std::vector<cytnx_uint64> &invmapper_L,
                                               const std::vector<cytnx_uint64> &invmapper_R) {
        const cytnx_uint64 len = Lt._impl->storage()._impl->size();
        const bool rhs_is_scalar = (Rt._impl->storage()._impl->size() == 1);

        std::visit(
          [&](auto *lptr) {
            using TL = std::remove_pointer_t<decltype(lptr)>;
            static_assert(!std::is_same_v<TL, void>);

            std::visit(
              [&](auto *rptr) {
                using TR = std::remove_pointer_t<decltype(rptr)>;
                static_assert(!std::is_same_v<TR, void>);
                ApplyInplaceArithmeticKernel<op_code, TL, TR>(lptr, rptr, len, rhs_is_scalar, shape,
                                                               invmapper_L, invmapper_R);
              },
              Rt.ptr());
          },
          Lt.ptr());
      }

    }  // namespace detail
  }  // namespace linalg
}  // namespace cytnx

#endif  // CYTNX_LINALG_IARITHMETIC_VISIT_HPP_
