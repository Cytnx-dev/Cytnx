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
        // surviving axes are symmetric in (ax1, ax2).
        const auto &input_shape = Tn.shape();
        const std::vector<cytnx_uint64> input_strides = Tn.strides();
        const cytnx_uint64 diagonal_length = input_shape[ax1];
        const cytnx_uint64 diagonal_stride = input_strides[ax1] + input_strides[ax2];

        // Build the reduced output shape and the matching per-surviving-axis input
        // strides in a single pass over the surviving axes (no separate surviving-
        // axis index list). The output indices and the per-surviving-axis strides
        // are kept signed so the odometer wrap-around step
        // (diagonal_start_offset -= (output_shape[axis] - 1) * stride) stays
        // arithmetic on a single signed type without casts.
        std::vector<cytnx_int64> output_shape;  // for Tensor::reshape_
        std::vector<cytnx_int64> surviving_input_stride;
        for (cytnx_uint64 axis = 0; axis < input_shape.size(); ++axis) {
          if (axis != ax1 && axis != ax2) {
            output_shape.push_back(static_cast<cytnx_int64>(input_shape[axis]));
            surviving_input_stride.push_back(static_cast<cytnx_int64>(input_strides[axis]));
          }
        }
        const cytnx_int64 surviving_rank = static_cast<cytnx_int64>(surviving_input_stride.size());
        cytnx_int64 output_size = 1;
        for (auto dim : output_shape) output_size *= dim;
        const bool output_is_scalar = surviving_rank == 0;

        // Fill a flat result Storage, then compose the output Tensor from it; the
        // 2D trace produces a single element, higher-rank traces produce one
        // element per surviving-axis multi-index.
        Storage output_storage(
          output_is_scalar ? cytnx_uint64{1} : static_cast<cytnx_uint64>(output_size), Tn.dtype(),
          Tn.device());
        if (diagonal_length == 0 || output_size == 0) {
          output_storage.set_zeros();
          Tensor out = Tensor::from_storage(output_storage);
          if (!output_is_scalar) out.reshape_(output_shape);
          return out;
        }

        const cytnx_uint64 diagonal_span = (diagonal_length - 1) * diagonal_stride + 1;
        const T *input_data = Tn.storage().data<T>();
        T *output_data = output_storage.data<T>();

        if (output_is_scalar) {
          output_data[0] =
            PairwiseSum(std::span<const T>(input_data, diagonal_span) | stride(diagonal_stride));
          return Tensor::from_storage(output_storage);
        }

        // Walk the output elements in row-major order, carrying the diagonal's
        // start offset in the input on an odometer: each step bumps the last
        // surviving-axis index (carrying into earlier axes on wrap) and adjusts the
        // offset by the affected axes' input strides. This avoids the per-element
        // division and modulo of decoding the flat index, and needs no precomputed
        // row-major accumulators.
        std::vector<cytnx_int64> surviving_index(static_cast<std::size_t>(surviving_rank), 0);
        cytnx_int64 diagonal_start_offset = 0;
        for (cytnx_int64 output_index = 0; output_index < output_size; ++output_index) {
          output_data[output_index] =
            PairwiseSum(std::span<const T>(input_data + diagonal_start_offset, diagonal_span) |
                        stride(diagonal_stride));
          for (cytnx_int64 axis = surviving_rank - 1; axis >= 0; --axis) {
            if (++surviving_index[axis] < output_shape[axis]) {
              diagonal_start_offset += surviving_input_stride[axis];
              break;
            }
            surviving_index[axis] = 0;
            diagonal_start_offset -= (output_shape[axis] - 1) * surviving_input_stride[axis];
          }
        }
        Tensor out = Tensor::from_storage(output_storage);
        out.reshape_(output_shape);
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
