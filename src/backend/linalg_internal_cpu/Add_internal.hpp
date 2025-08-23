#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_ADD_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_ADD_INTERNAL_H_

#include <assert.h>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"
#include "utils/utils.hpp"

namespace {
  template <typename TL, typename TR>
  static inline constexpr bool leftTypeIsHigher() {
    return cytnx::type_priority<TL>::value > cytnx::type_priority<TR>::value;
  }

  template <typename TLin, typename TRin>
  static inline void AddInternalImpl(boost::intrusive_ptr<cytnx::Storage_base> &out,
                                     boost::intrusive_ptr<cytnx::Storage_base> &Lin,
                                     boost::intrusive_ptr<cytnx::Storage_base> &Rin,
                                     const cytnx::cytnx_uint64 &len,
                                     const std::vector<cytnx::cytnx_uint64> &shape,
                                     const std::vector<cytnx::cytnx_uint64> &invmapper_L,
                                     const std::vector<cytnx::cytnx_uint64> &invmapper_R) {
    TLin *_out = reinterpret_cast<TLin *>(out->data());
    const TLin *_Lin = reinterpret_cast<const TLin *>(Lin->data());
    const TRin *_Rin = reinterpret_cast<const TRin *>(Rin->data());

    if (Lin->size() == 1) {
      for (cytnx::cytnx_uint64 i = 0; i < len; i++) {
        _out[i] = static_cast<TLin>(static_cast<TLin>(_Lin[0]) + static_cast<TLin>(_Rin[i]));
      }
    } else if (Rin->size() == 1) {
      for (cytnx::cytnx_uint64 i = 0; i < len; i++) {
        _out[i] = static_cast<TLin>(static_cast<TLin>(_Lin[i]) + static_cast<TLin>(_Rin[0]));
      }
    } else {
      if (shape.size() == 0) {
        // Contiguous case
        for (cytnx::cytnx_uint64 i = 0; i < len; i++) {
          _out[i] = static_cast<TLin>(static_cast<TLin>(_Lin[i]) + static_cast<TLin>(_Rin[i]));
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
          _out[i] =
            static_cast<TLin>(static_cast<TLin>(_Lin[idx_L]) + static_cast<TLin>(_Rin[idx_R]));
        }
      }
    }
  }
}  // namespace

namespace cytnx {
  namespace linalg_internal {
    template <typename TLin, typename TRin>
    inline void AddInternalInterface(boost::intrusive_ptr<Storage_base> &out,
                                     boost::intrusive_ptr<Storage_base> &Lin,
                                     boost::intrusive_ptr<Storage_base> &Rin,
                                     const cytnx_uint64 &len,
                                     const std::vector<cytnx_uint64> &shape,
                                     const std::vector<cytnx_uint64> &invmapper_L,
                                     const std::vector<cytnx_uint64> &invmapper_R) {
      if constexpr (::leftTypeIsHigher<TLin, TRin>()) {
        // Left type is higher, use normal order
        ::AddInternalImpl<TLin, TRin>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
      } else {
        // Right type is higher, swap the operands and mappers
        ::AddInternalImpl<TRin, TLin>(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
      }
    }
  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_ADD_INTERNAL_H_
