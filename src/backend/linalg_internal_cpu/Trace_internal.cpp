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
      Tensor TraceImpl(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
        // Trace() validates upstream that ax1 != ax2 and shape[ax1] == shape[ax2],
        // so their order is irrelevant: the diagonal stride and the set of
        // remaining axes are symmetric in (ax1, ax2).
        const auto &shape_in = Tn.shape();
        const std::vector<cytnx_uint64> strides = Tn.strides();
        const cytnx_uint64 n_diag = shape_in[ax1];
        const cytnx_uint64 diag_stride = strides[ax1] + strides[ax2];

        // Build the reduced output shape and the matching per-output-axis input
        // strides in a single pass over the surviving axes (no separate
        // remaining-axis index list).
        std::vector<cytnx_int64> out_shape;  // reduced shape, for reshape_
        std::vector<cytnx_uint64> out_strides;  // input stride of each surviving axis
        for (cytnx_uint64 i = 0; i < shape_in.size(); ++i) {
          if (i != ax1 && i != ax2) {
            out_shape.push_back(static_cast<cytnx_int64>(shape_in[i]));
            out_strides.push_back(strides[i]);
          }
        }
        const cytnx_uint64 out_rank = out_strides.size();
        cytnx_uint64 n_elem = 1;
        for (auto dim : out_shape) n_elem *= static_cast<cytnx_uint64>(dim);
        const bool is_2d = out_rank == 0;

        // Fill a flat result Storage, then compose the output Tensor from it; the
        // 2D trace produces a single element, the ND trace one element per
        // remaining-rank multi-index.
        Storage out_storage(is_2d ? cytnx_uint64{1} : n_elem, Tn.dtype(), Tn.device());
        if (n_diag == 0 || n_elem == 0) {
          out_storage.set_zeros();
          Tensor out = Tensor::from_storage(out_storage);
          if (!is_2d) out.reshape_(out_shape);
          return out;
        }

        const cytnx_uint64 extent = (n_diag - 1) * diag_stride + 1;
        const T *data = Tn.storage().data<T>();
        T *out_data = out_storage.data<T>();

        if (is_2d) {
          out_data[0] = PairwiseSum(std::span<const T>(data, extent) | stride(diag_stride));
          return Tensor::from_storage(out_storage);
        }

        // Walk the output elements in row-major order, carrying the input base
        // offset on an odometer: each step bumps the last axis index (carrying into
        // earlier axes on wrap) and adjusts base by the affected axes' strides. This
        // avoids the per-element division and modulo of decoding the flat index, and
        // needs no precomputed row-major accumulators.
        std::vector<cytnx_uint64> index(out_rank, 0);
        cytnx_uint64 base = 0;
        for (cytnx_uint64 i = 0; i < n_elem; ++i) {
          out_data[i] = PairwiseSum(std::span<const T>(data + base, extent) | stride(diag_stride));
          for (cytnx_uint64 x = out_rank; x-- > 0;) {
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
