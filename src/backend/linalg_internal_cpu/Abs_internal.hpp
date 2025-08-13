#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_ABS_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_ABS_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    template <typename TIn, typename TOut>
    void AbsInternalImpl(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem) {
      TOut *_out = reinterpret_cast<TOut *>(out->data());
      const TIn *_ten = reinterpret_cast<const TIn *>(ten->data());

      if constexpr (is_complex_v<TIn>) {
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          _out[n] = static_cast<TOut>(std::abs(_ten[n]));
        }
      } else if constexpr (std::is_same_v<TIn, cytnx_double>) {
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          _out[n] = static_cast<TOut>(std::fabs(_ten[n]));
        }
      } else if constexpr (std::is_same_v<TIn, cytnx_float>) {
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          _out[n] = static_cast<TOut>(std::fabs(_ten[n]));
        }
      } else if constexpr (std::is_same_v<TIn, cytnx_int64>) {
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          _out[n] = static_cast<TOut>(std::llabs(_ten[n]));
        }
      } else if constexpr (std::is_same_v<TIn, cytnx_int32>) {
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          _out[n] = static_cast<TOut>(std::labs(_ten[n]));
        }
      } else if constexpr (std::is_same_v<TIn, cytnx_int16>) {
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          _out[n] = static_cast<TOut>(std::fabs(static_cast<double>(_ten[n])));
        }
      } else {
        cytnx_error_msg(
          true, "[ERROR][AbsInternalImpl] Unsupported type combination for abs operation.%s", "\n");
      }
    }

  }  // namespace linalg_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_ABS_INTERNAL_H_
