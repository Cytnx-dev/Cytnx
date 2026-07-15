#ifndef CYTNX_LINALG_IARITHMETIC_VISIT_HPP_
#define CYTNX_LINALG_IARITHMETIC_VISIT_HPP_

#include <type_traits>
#include <vector>

#include "Tensor.hpp"
#include "utils/utils.hpp"

namespace cytnx {
  namespace linalg {
    namespace detail {

      // op_code: 0=Add, 1=Mul, 2=Sub, 3=Div (true division).
      template <char op_code, typename TO, typename TL, typename TR>
      inline TO ApplyInplaceArithmeticOp(const TL &lhs, const TR &rhs) {
        if constexpr (!cytnx::is_complex_v<TO> &&
                      (cytnx::is_complex_v<TL> || cytnx::is_complex_v<TR>)) {
          cytnx_error_msg(true, "[ERROR][inplace arithmetic] Cannot narrow complex into real%s",
                          "\n");
          return TO{};
        } else {
          if constexpr (op_code == 0) {
            return static_cast<TO>(static_cast<TO>(lhs) + static_cast<TO>(rhs));
          } else if constexpr (op_code == 1) {
            return static_cast<TO>(static_cast<TO>(lhs) * static_cast<TO>(rhs));
          } else if constexpr (op_code == 2) {
            return static_cast<TO>(static_cast<TO>(lhs) - static_cast<TO>(rhs));
          } else {
            static_assert(op_code == 3);
            return static_cast<TO>(static_cast<TO>(lhs) / static_cast<TO>(rhs));
          }
        }
      }

      template <char op_code, typename TO, typename TL, typename TR>
      inline void ApplyInplaceArithmeticKernel(TO *out, const TL *lhs, const TR *rhs,
                                               const cytnx_uint64 &len, const bool &rhs_is_scalar,
                                               const std::vector<cytnx_uint64> &shape,
                                               const std::vector<cytnx_uint64> &invmapper_L,
                                               const std::vector<cytnx_uint64> &invmapper_R) {
        if (rhs_is_scalar) {
          for (cytnx_uint64 i = 0; i < len; i++) {
            out[i] = ApplyInplaceArithmeticOp<op_code, TO>(lhs[i], rhs[0]);
          }
          return;
        }

        if (shape.empty()) {
          for (cytnx_uint64 i = 0; i < len; i++) {
            out[i] = ApplyInplaceArithmeticOp<op_code, TO>(lhs[i], rhs[i]);
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
          // In-place semantics: the result lands at lhs's PHYSICAL position
          // idx_L (out either aliases lhs's buffer, or is a fresh promoted
          // buffer that Lt's unchanged meta/invmapper will address the same
          // way) -- NOT at the logical linear index i.
          out[idx_L] = ApplyInplaceArithmeticOp<op_code, TO>(lhs[idx_L], rhs[idx_R]);
        }
      }

      // Output-dtype rule per op_code: Div (3) uses true-division
      // (make_floating_point_t<type_promote_t<TL,TR>>); everything else uses
      // plain type_promote_t<TL,TR> (#941's stated default rule).
      template <char op_code, typename TL, typename TR>
      struct InplaceOutputType {
        using type = Type_class::type_promote_t<TL, TR>;
      };
      template <typename TL, typename TR>
      struct InplaceOutputType<3, TL, TR> {
        using type = Type_class::make_floating_point_t<Type_class::type_promote_t<TL, TR>>;
      };
      template <char op_code, typename TL, typename TR>
      using InplaceOutputType_t = typename InplaceOutputType<op_code, TL, TR>::type;

      // Dispatch in-place arithmetic on CPU with correct dtype promotion: if
      // the promoted output type differs from Lt's current storage dtype,
      // allocate new storage of the promoted type and replace Lt's storage
      // (matching the equivalent out-of-place operation's output type and
      // Python in-place semantics) -- rather than truncating results into the
      // original narrower/incompatible storage (#941's core bug: a dispatch
      // table could select a kernel whose C++ output type was e.g. double,
      // while the actual output storage object was still
      // StorageImplementation<int16_t>).
      template <char op_code>
      inline void DispatchInplaceArithmeticCPU(Tensor &Lt, const Tensor &Rt,
                                               const std::vector<cytnx_uint64> &shape,
                                               const std::vector<cytnx_uint64> &invmapper_L,
                                               const std::vector<cytnx_uint64> &invmapper_R) {
        const cytnx_uint64 len = Lt._impl->storage()._impl->size();
        const bool rhs_is_scalar = is_singleton_tensor(Rt);
        const int device = Lt.device();
        // A rank-0 RHS is a python-scalar wrapper (Tensor::operator op=(scalar)
        // routes through scalar_as_rank0_tensor). Per the #1015 ruling affirming
        // #980, `tensor op= python-scalar` follows numpy weak-scalar semantics:
        // it PRESERVES the LHS dtype (the scalar is cast into TL) rather than
        // promoting. A rank>=1 RHS is a genuine tensor and promotes (#941).
        // Complex-into-real is still rejected by ApplyInplaceArithmeticOp.
        const bool weak_scalar_rhs = (Rt.rank() == 0);

        std::visit(
          [&](auto lhs_impl, auto rhs_impl) {
            using TL = storage_value_t<decltype(lhs_impl)>;
            using TR = storage_value_t<decltype(rhs_impl)>;
            // weak-scalar RHS keeps TL; a genuine tensor RHS promotes.
            using TO = InplaceOutputType_t<op_code, TL, TR>;

            // storage_as_type_or_replace<TO> may replace Lt._impl->storage()._impl
            // (e.g. when TO != TL, the promoted output dtype differs from Lt's
            // current storage dtype). That is safe here: lhs_impl was already
            // captured above as an owning intrusive_ptr<StorageImplementation<TL>>
            // to the ORIGINAL buffer before this call, so lhs_impl->data() stays
            // valid and points at the pre-replacement data for the kernel call
            // below, exactly as #941 describes for the aliased in-place case.
            if (weak_scalar_rhs) {
              // Weak-scalar output: treat the RHS dtype as TL, so Add/Sub/Mul
              // keep TL (numpy weak-scalar, #980) while Div still follows #941
              // true-division (make_floating_point(TL): Int64 /= 2.0 -> Double,
              // Float /= 3.0 -> Float). Complex-into-real is still rejected by
              // ApplyInplaceArithmeticOp below.
              using TO_weak = InplaceOutputType_t<op_code, TL, TL>;
              auto out_impl = storage_as_type_or_replace<TO_weak>(Lt._impl->storage(), len, device);
              ApplyInplaceArithmeticKernel<op_code, TO_weak, TL, TR>(
                out_impl->data(), lhs_impl->data(), rhs_impl->data(), len, /*rhs_is_scalar=*/true,
                shape, invmapper_L, invmapper_R);
            } else {
              auto out_impl = storage_as_type_or_replace<TO>(Lt._impl->storage(), len, device);
              ApplyInplaceArithmeticKernel<op_code, TO, TL, TR>(
                out_impl->data(), lhs_impl->data(), rhs_impl->data(), len, rhs_is_scalar, shape,
                invmapper_L, invmapper_R);
            }
          },
          Lt._impl->storage().as_storage_variant(), Rt._impl->storage().as_storage_variant());
      }

    }  // namespace detail
  }  // namespace linalg
}  // namespace cytnx

#endif  // CYTNX_LINALG_IARITHMETIC_VISIT_HPP_
