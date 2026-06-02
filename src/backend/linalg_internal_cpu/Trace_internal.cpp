#include "Trace_internal.hpp"
#include "Tensor.hpp"
#include "backend/Storage.hpp"
#include "cytnx_error.hpp"

#include "backend/linalg_internal_cpu/pairwise_sum.hpp"
#include "backend/linalg_internal_cpu/stride_view.hpp"

#include <algorithm>
#include <span>
#include <vector>

namespace cytnx {
  namespace linalg_internal {
    namespace {

      template <class T>
      Tensor TraceImpl(const Tensor &Tn, cytnx_uint64 a1, cytnx_uint64 a2) {
        const cytnx_uint64 ax1 = std::min(a1, a2);
        const cytnx_uint64 ax2 = std::max(a1, a2);
        const auto &shape_in = Tn.shape();
        const cytnx_uint64 Ndiag = shape_in[ax1];

        std::vector<cytnx_int64> out_shape;
        std::vector<cytnx_uint64> remain_rank_id;
        for (cytnx_uint64 i = 0; i < shape_in.size(); ++i) {
          if (i != ax1 && i != ax2) {
            out_shape.push_back(static_cast<cytnx_int64>(shape_in[i]));
            remain_rank_id.push_back(i);
          }
        }
        cytnx_uint64 Nelem = 1;
        for (auto d : out_shape) Nelem *= static_cast<cytnx_uint64>(d);
        const bool is_2d = out_shape.empty();

        // Fill a flat result Storage, then compose the output Tensor from it; the
        // 2D trace produces a single element, the ND trace one element per
        // remaining-rank multi-index.
        Storage out_storage(is_2d ? cytnx_uint64{1} : Nelem, Tn.dtype(), Tn.device());
        if (Ndiag == 0 || Nelem == 0) {
          out_storage.set_zeros();
          Tensor out = Tensor::from_storage(out_storage);
          if (!is_2d) out.reshape_(out_shape);
          return out;
        }

        const std::vector<cytnx_uint64> strides = Tn.strides();
        const cytnx_uint64 diag_stride = strides[ax1] + strides[ax2];
        const cytnx_uint64 extent = (Ndiag - 1) * diag_stride + 1;
        const T *data = Tn.storage().data<T>();
        T *out_data = out_storage.data<T>();

        if (is_2d) {
          out_data[0] = PairwiseSum(std::span<const T>(data, extent) | stride(diag_stride));
          return Tensor::from_storage(out_storage);
        }

        // Input stride for each surviving (output) axis, so the hot loop indexes a
        // flat array instead of going through remain_rank_id on every step.
        std::vector<cytnx_uint64> out_strides(out_shape.size());
        for (cytnx_uint64 x = 0; x < out_shape.size(); ++x)
          out_strides[x] = strides[remain_rank_id[x]];

        // Walk the output elements in row-major order, carrying the input base
        // offset on an odometer: each step bumps the last axis index (carrying into
        // earlier axes on wrap) and adjusts base by the affected axes' strides. This
        // avoids the per-element division and modulo of decoding the flat index, and
        // needs no precomputed row-major accumulators.
        std::vector<cytnx_uint64> index(out_shape.size(), 0);
        cytnx_uint64 base = 0;
        for (cytnx_uint64 i = 0; i < Nelem; ++i) {
          out_data[i] = PairwiseSum(std::span<const T>(data + base, extent) | stride(diag_stride));
          for (cytnx_uint64 x = out_shape.size(); x-- > 0;) {
            if (++index[x] < static_cast<cytnx_uint64>(out_shape[x])) {
              base += out_strides[x];
              break;
            }
            index[x] = 0;
            base -= (static_cast<cytnx_uint64>(out_shape[x]) - 1) * out_strides[x];
          }
        }
        Tensor out = Tensor::from_storage(out_storage);
        out.reshape_(out_shape);
        return out;
      }

    }  // namespace

    Tensor Trace_internal_cd(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImpl<cytnx_complex128>(Tn, ax1, ax2);
    }
    Tensor Trace_internal_cf(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImpl<cytnx_complex64>(Tn, ax1, ax2);
    }
    Tensor Trace_internal_d(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImpl<cytnx_double>(Tn, ax1, ax2);
    }
    Tensor Trace_internal_f(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImpl<cytnx_float>(Tn, ax1, ax2);
    }
    Tensor Trace_internal_u64(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImpl<cytnx_uint64>(Tn, ax1, ax2);
    }
    Tensor Trace_internal_i64(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImpl<cytnx_int64>(Tn, ax1, ax2);
    }
    Tensor Trace_internal_u32(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImpl<cytnx_uint32>(Tn, ax1, ax2);
    }
    Tensor Trace_internal_i32(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImpl<cytnx_int32>(Tn, ax1, ax2);
    }
    Tensor Trace_internal_u16(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImpl<cytnx_uint16>(Tn, ax1, ax2);
    }
    Tensor Trace_internal_i16(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImpl<cytnx_int16>(Tn, ax1, ax2);
    }

    Tensor Trace_internal_b(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      cytnx_error_msg(true, "[internal][Trace] bool is not available. %s", "\n");
      return Tensor();
    }

  }  // namespace linalg_internal

}  // namespace cytnx
