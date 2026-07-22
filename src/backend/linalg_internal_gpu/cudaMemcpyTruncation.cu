#include <stdlib.h>
#include <stdio.h>

#include <type_traits>
#include <variant>
#include <vector>

#include "cudaMemcpyTruncation.hpp"
#include "Generator.hpp"

#ifdef UNI_GPU
  #define HANDLE_CUDA_ERROR(x)                                                    \
    {                                                                             \
      const auto err = x;                                                         \
      if (err != cudaSuccess) {                                                   \
        printf("CUDA Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); \
        fflush(stdout);                                                           \
      }                                                                           \
    };
#endif

namespace cytnx {
  namespace linalg_internal {

#ifdef UNI_GPU
    // GPU counterpart of linalg_internal::memcpyTruncation (see the CPU implementation for details)
    // The singular values S are scanned on a host copy, and the actual
    // S / U / vT / terr copies are device-to-device cudaMemcpy dispatched on the native gpu
    // pointer via std::visit.
    void cudaMemcpyTruncation(std::vector<Tensor> &tens, cytnx_uint64 keepdim, double err,
                              bool is_U, bool is_vT, unsigned int return_err, cytnx_uint64 mindim) {
      // at least one singular value is always kept
      if (mindim < 1) mindim = 1;
      const cytnx_uint64 nums = tens[0].storage().size();
      cytnx_uint64 trunc_dim = (nums < keepdim) ? nums : keepdim;

      if (nums == 0) {
        if (return_err == 1) {
          tens.push_back(zeros({}, tens[0].dtype(), tens[0].device()));
        } else if (return_err) {
          tens.push_back(zeros({0}, tens[0].dtype(), tens[0].device()));
        }
        return;
      }

      // --- determine the truncation dimension from a HOST copy of the (real) singular values S ---
      Tensor S_host = tens[0].to(Device.cpu).contiguous();
      std::visit(
        [&](auto sptr) {
          using T = std::remove_pointer_t<decltype(sptr)>;
          if constexpr (std::is_floating_point_v<T>) {
            for (cytnx_int64 i = trunc_dim - 1; i >= 0; --i) {
              if (sptr[i] < err && trunc_dim - 1 >= mindim) {
                trunc_dim--;
              } else {
                break;
              }
            }
            if (trunc_dim == 0) trunc_dim = 1;
          } else {
            cytnx_error_msg(true,
                            "[cudaMemcpyTruncation] singular values S must be a real "
                            "floating-point type, got %s.\n",
                            Type.getname(tens[0].dtype()).c_str());
          }
        },
        S_host.ptr());

      if (trunc_dim == nums) {
        // nothing to truncate; no discarded values, so the truncation error is zero.
        if (return_err == 1) {
          tens.push_back(zeros({}, tens[0].dtype(), tens[0].device()));
        } else if (return_err) {
          tens.push_back(zeros({0}, tens[0].dtype(), tens[0].device()));
        }
        return;
      }

      // --- truncate S and build terr on the device (native real type) ---
      Tensor terr;
      Tensor S = tens[0].contiguous();
      std::visit(
        [&](auto sptr) {
          using T = std::remove_pointer_t<decltype(sptr)>;
          if constexpr (std::is_floating_point_v<T>) {
            Tensor newS({trunc_dim}, S.dtype(), S.device());
            HANDLE_CUDA_ERROR(cudaMemcpy(newS.gpu_ptr_as<T>(), sptr, trunc_dim * sizeof(T),
                                         cudaMemcpyDeviceToDevice));
            if (return_err == 1) {
              terr = Tensor({}, S.dtype(), S.device());
              HANDLE_CUDA_ERROR(cudaMemcpy(terr.gpu_ptr_as<T>(), sptr + trunc_dim, sizeof(T),
                                           cudaMemcpyDeviceToDevice));
            } else if (return_err) {
              const cytnx_uint64 discarded_dim = nums - trunc_dim;
              terr = Tensor({discarded_dim}, S.dtype(), S.device());
              HANDLE_CUDA_ERROR(cudaMemcpy(terr.gpu_ptr_as<T>(), sptr + trunc_dim,
                                           discarded_dim * sizeof(T), cudaMemcpyDeviceToDevice));
            }
            tens[0] = newS;
          }
        },
        S.gpu_ptr());

      // --- U / vT: located positionally in the packed vector ([S, U?, vT?]); device copies
      // dispatched on the matrix dtype via std::visit ---
      cytnx_uint64 t = 1;
      if (is_U) {
        Tensor src = tens[t].contiguous();
        cytnx_uint64 rows = src.shape()[0], cols = src.shape()[1];
        Tensor newU({rows, trunc_dim}, src.dtype(), src.device());
        std::visit(
          [&](auto sptr) {
            using T = std::remove_pointer_t<decltype(sptr)>;
            T *dst = newU.gpu_ptr_as<T>();
            // keep the first trunc_dim columns of each row; row-major, so copy row by row
            for (cytnx_uint64 i = 0; i < rows; ++i) {
              HANDLE_CUDA_ERROR(cudaMemcpy(dst + i * trunc_dim, sptr + i * cols,
                                           trunc_dim * sizeof(T), cudaMemcpyDeviceToDevice));
            }
          },
          src.gpu_ptr());
        tens[t] = newU;
        t++;
      }
      if (is_vT) {
        Tensor src = tens[t].contiguous();
        cytnx_uint64 cols = src.shape()[1];
        Tensor newvT({trunc_dim, cols}, src.dtype(), src.device());
        std::visit(
          [&](auto sptr) {
            using T = std::remove_pointer_t<decltype(sptr)>;
            // keeping the first trunc_dim rows is a single contiguous prefix copy
            HANDLE_CUDA_ERROR(cudaMemcpy(newvT.gpu_ptr_as<T>(), sptr, trunc_dim * cols * sizeof(T),
                                         cudaMemcpyDeviceToDevice));
          },
          src.gpu_ptr());
        tens[t] = newvT;
      }

      if (return_err) tens.push_back(terr);
    }
#endif
  }  // namespace linalg_internal
}  // namespace cytnx
