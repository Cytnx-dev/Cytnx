#include "Trace_internal.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"

#include "backend/linalg_internal_cpu/pairwise_sum.hpp"
#include "backend/linalg_internal_cpu/stride_view.hpp"
#include "utils/utils.hpp"

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

        Tensor out = Tensor({is_2d ? cytnx_uint64{1} : Nelem}, Tn.dtype(), Tn.device());
        out.storage().set_zeros();
        if (Ndiag == 0 || Nelem == 0) {
          if (!is_2d) out.reshape_(out_shape);
          return out;
        }

        const std::vector<cytnx_uint64> strides = Tn.strides();
        const cytnx_uint64 diag_stride = strides[ax1] + strides[ax2];
        const cytnx_uint64 extent = (Ndiag - 1) * diag_stride + 1;
        const T *data = Tn.storage().data<T>();

        if (is_2d) {
          out.storage().at<T>(0) =
            PairwiseSum(std::span<const T>(data, extent) | stride(diag_stride));
          return out;
        }

        std::vector<cytnx_uint64> accu(out_shape.size(), 1);
        for (int i = static_cast<int>(out_shape.size()) - 1; i > 0; --i)
          accu[i - 1] = accu[i] * static_cast<cytnx_uint64>(out_shape[i]);

        T *out_data = out.storage().data<T>();
        for (cytnx_uint64 i = 0; i < Nelem; ++i) {
          cytnx_uint64 tmp = i, base = 0;
          for (cytnx_uint64 x = 0; x < out_shape.size(); ++x) {
            base += (tmp / accu[x]) * strides[remain_rank_id[x]];
            tmp %= accu[x];
          }
          out_data[i] = PairwiseSum(std::span<const T>(data + base, extent) | stride(diag_stride));
        }
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
