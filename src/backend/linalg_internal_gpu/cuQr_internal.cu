#include "cuQr_internal.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <variant>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "backend/utils_internal_gpu/cuLibraryHandle_gpu.hpp"
#include "backend/utils_internal_gpu/cuScopedResource_gpu.hpp"

namespace cytnx {
  namespace linalg_internal {

    namespace {

      template <class T>
      concept CusolverQrScalar = std::same_as<T, cytnx_float> || std::same_as<T, cytnx_double> ||
        std::same_as<T, cytnx_cuda_complex64> || std::same_as<T, cytnx_cuda_complex128>;

      // dst (column-major rows x cols) = src (row-major rows x cols).
      template <class T>
      __global__ void cu_qr_to_col_major_kernel(T *dst, const T *src, std::uint64_t rows,
                                                std::uint64_t cols) {
        const std::uint64_t idx = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx < rows * cols) {
          const std::uint64_t i = idx / cols;
          const std::uint64_t j = idx % cols;
          dst[j * rows + i] = src[idx];
        }
      }

      // dst (row-major rows x cols) = the leading rows x cols part of src (column-major, leading
      // dimension ld). With upper_only, entries below the main diagonal are written as zero.
      template <class T>
      __global__ void cu_qr_to_row_major_kernel(T *dst, const T *src, std::uint64_t rows,
                                                std::uint64_t cols, std::uint64_t ld,
                                                bool upper_only) {
        const std::uint64_t idx = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx < rows * cols) {
          const std::uint64_t i = idx / cols;
          const std::uint64_t j = idx % cols;
          dst[idx] = (upper_only && j < i) ? T(0) : src[j * ld + i];
        }
      }

      constexpr unsigned int kThreadsPerBlock = 256;

      unsigned int block_count(std::uint64_t elem_count) {
        const std::uint64_t blocks = (elem_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
        cytnx_error_msg(blocks > static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
                        "[cuQr] matrix with %llu elements is too large for a single CUDA kernel "
                        "launch.\n",
                        static_cast<unsigned long long>(elem_count));
        return static_cast<unsigned int>(blocks);
      }

      // cuSOLVER takes the cuComplex ABI types; device code uses cuda::std::complex. The two are
      // layout-compatible, so the cast happens here, at the library-call boundary.
      cytnx_float *cusolver_ptr(cytnx_float *ptr) { return ptr; }
      cytnx_double *cusolver_ptr(cytnx_double *ptr) { return ptr; }
      cuComplex *cusolver_ptr(cytnx_cuda_complex64 *ptr) {
        return reinterpret_cast<cuComplex *>(ptr);
      }
      cuDoubleComplex *cusolver_ptr(cytnx_cuda_complex128 *ptr) {
        return reinterpret_cast<cuDoubleComplex *>(ptr);
      }

      cusolverStatus_t geqrf_buffer_size(cusolverDnHandle_t handle, int m, int n, cytnx_float *a,
                                         int lda, int *lwork) {
        return cusolverDnSgeqrf_bufferSize(handle, m, n, a, lda, lwork);
      }
      cusolverStatus_t geqrf_buffer_size(cusolverDnHandle_t handle, int m, int n, cytnx_double *a,
                                         int lda, int *lwork) {
        return cusolverDnDgeqrf_bufferSize(handle, m, n, a, lda, lwork);
      }
      cusolverStatus_t geqrf_buffer_size(cusolverDnHandle_t handle, int m, int n, cuComplex *a,
                                         int lda, int *lwork) {
        return cusolverDnCgeqrf_bufferSize(handle, m, n, a, lda, lwork);
      }
      cusolverStatus_t geqrf_buffer_size(cusolverDnHandle_t handle, int m, int n,
                                         cuDoubleComplex *a, int lda, int *lwork) {
        return cusolverDnZgeqrf_bufferSize(handle, m, n, a, lda, lwork);
      }

      cusolverStatus_t geqrf(cusolverDnHandle_t handle, int m, int n, cytnx_float *a, int lda,
                             cytnx_float *tau, cytnx_float *work, int lwork, int *dev_info) {
        return cusolverDnSgeqrf(handle, m, n, a, lda, tau, work, lwork, dev_info);
      }
      cusolverStatus_t geqrf(cusolverDnHandle_t handle, int m, int n, cytnx_double *a, int lda,
                             cytnx_double *tau, cytnx_double *work, int lwork, int *dev_info) {
        return cusolverDnDgeqrf(handle, m, n, a, lda, tau, work, lwork, dev_info);
      }
      cusolverStatus_t geqrf(cusolverDnHandle_t handle, int m, int n, cuComplex *a, int lda,
                             cuComplex *tau, cuComplex *work, int lwork, int *dev_info) {
        return cusolverDnCgeqrf(handle, m, n, a, lda, tau, work, lwork, dev_info);
      }
      cusolverStatus_t geqrf(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *a, int lda,
                             cuDoubleComplex *tau, cuDoubleComplex *work, int lwork,
                             int *dev_info) {
        return cusolverDnZgeqrf(handle, m, n, a, lda, tau, work, lwork, dev_info);
      }

      // "orgqr" stands for both orgqr (real) and ungqr (complex): form the explicit m x n matrix
      // with orthonormal columns from the first k reflectors. Requires m >= n >= k >= 0.
      cusolverStatus_t orgqr_buffer_size(cusolverDnHandle_t handle, int m, int n, int k,
                                         const cytnx_float *a, int lda, const cytnx_float *tau,
                                         int *lwork) {
        return cusolverDnSorgqr_bufferSize(handle, m, n, k, a, lda, tau, lwork);
      }
      cusolverStatus_t orgqr_buffer_size(cusolverDnHandle_t handle, int m, int n, int k,
                                         const cytnx_double *a, int lda, const cytnx_double *tau,
                                         int *lwork) {
        return cusolverDnDorgqr_bufferSize(handle, m, n, k, a, lda, tau, lwork);
      }
      cusolverStatus_t orgqr_buffer_size(cusolverDnHandle_t handle, int m, int n, int k,
                                         const cuComplex *a, int lda, const cuComplex *tau,
                                         int *lwork) {
        return cusolverDnCungqr_bufferSize(handle, m, n, k, a, lda, tau, lwork);
      }
      cusolverStatus_t orgqr_buffer_size(cusolverDnHandle_t handle, int m, int n, int k,
                                         const cuDoubleComplex *a, int lda,
                                         const cuDoubleComplex *tau, int *lwork) {
        return cusolverDnZungqr_bufferSize(handle, m, n, k, a, lda, tau, lwork);
      }

      cusolverStatus_t orgqr(cusolverDnHandle_t handle, int m, int n, int k, cytnx_float *a,
                             int lda, const cytnx_float *tau, cytnx_float *work, int lwork,
                             int *dev_info) {
        return cusolverDnSorgqr(handle, m, n, k, a, lda, tau, work, lwork, dev_info);
      }
      cusolverStatus_t orgqr(cusolverDnHandle_t handle, int m, int n, int k, cytnx_double *a,
                             int lda, const cytnx_double *tau, cytnx_double *work, int lwork,
                             int *dev_info) {
        return cusolverDnDorgqr(handle, m, n, k, a, lda, tau, work, lwork, dev_info);
      }
      cusolverStatus_t orgqr(cusolverDnHandle_t handle, int m, int n, int k, cuComplex *a, int lda,
                             const cuComplex *tau, cuComplex *work, int lwork, int *dev_info) {
        return cusolverDnCungqr(handle, m, n, k, a, lda, tau, work, lwork, dev_info);
      }
      cusolverStatus_t orgqr(cusolverDnHandle_t handle, int m, int n, int k, cuDoubleComplex *a,
                             int lda, const cuDoubleComplex *tau, cuDoubleComplex *work, int lwork,
                             int *dev_info) {
        return cusolverDnZungqr(handle, m, n, k, a, lda, tau, work, lwork, dev_info);
      }

      void check_dev_info(const int *dev_info, const char *routine) {
        int info = 0;
        checkCudaErrors(cudaMemcpy(&info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
        cytnx_error_msg(info != 0,
                        "[cuQr] cuSOLVER routine '%s' failed with devInfo = %d (a value of -i "
                        "means that the i-th argument is invalid).\n",
                        routine, info);
      }

      // in_ptr: row-major rows x cols. q_ptr: row-major rows x min(rows, cols). r_ptr: row-major
      // min(rows, cols) x cols.
      template <CusolverQrScalar T>
      void cu_qr_typed(const T *in_ptr, T *q_ptr, T *r_ptr, int rows, int cols) {
        const int rank_bound = std::min(rows, cols);
        const std::uint64_t rows64 = static_cast<std::uint64_t>(rows);
        const std::uint64_t cols64 = static_cast<std::uint64_t>(cols);
        const std::uint64_t rank_bound64 = static_cast<std::uint64_t>(rank_bound);

        cusolverDnHandle_t handle = utils_internal::get_cusolverdn_handle();

        // cuSOLVER is column-major and Cytnx storage is row-major, so factor an explicit
        // column-major copy; geqrf and orgqr both work in place on it.
        utils_internal::DeviceBuffer<T> factor(rows64 * cols64);
        cu_qr_to_col_major_kernel<<<block_count(rows64 * cols64), kThreadsPerBlock>>>(
          factor.get(), in_ptr, rows64, cols64);
        CYTNX_CHECK_CUDA_LAUNCH(cu_qr_to_col_major_kernel);

        // Scalar factors of the reflectors: produced by geqrf, consumed by orgqr / ungqr, and not
        // returned.
        utils_internal::DeviceBuffer<T> tau(rank_bound64);

        auto *factor_arg = cusolver_ptr(factor.get());
        auto *tau_arg = cusolver_ptr(tau.get());

        // Q is formed from the first rank_bound reflectors as a rows x rank_bound matrix. For a
        // wide input that is the leading rows x rows block of the factor, which satisfies the
        // m >= n >= k requirement of orgqr / ungqr.
        int geqrf_lwork = 0;
        int orgqr_lwork = 0;
        checkCudaErrors(geqrf_buffer_size(handle, rows, cols, factor_arg, rows, &geqrf_lwork));
        checkCudaErrors(orgqr_buffer_size(handle, rows, rank_bound, rank_bound, factor_arg, rows,
                                          tau_arg, &orgqr_lwork));
        const int lwork = std::max(geqrf_lwork, orgqr_lwork);
        cytnx_error_msg(lwork < 0, "[cuQr] cuSOLVER returned a negative workspace size %d.\n",
                        lwork);
        utils_internal::DeviceBuffer<T> work(static_cast<std::size_t>(lwork));
        auto *work_arg = cusolver_ptr(work.get());

        utils_internal::DeviceBuffer<int> dev_info(1);
        checkCudaErrors(cudaMemset(dev_info.get(), 0, sizeof(int)));

        checkCudaErrors(
          geqrf(handle, rows, cols, factor_arg, rows, tau_arg, work_arg, lwork, dev_info.get()));
        check_dev_info(dev_info.get(), "geqrf");

        // R is the upper triangle / trapezoid of the factor; the Householder vectors below the
        // diagonal are replaced by zeros.
        cu_qr_to_row_major_kernel<<<block_count(rank_bound64 * cols64), kThreadsPerBlock>>>(
          r_ptr, factor.get(), rank_bound64, cols64, rows64, true);
        CYTNX_CHECK_CUDA_LAUNCH(cu_qr_to_row_major_kernel);

        checkCudaErrors(orgqr(handle, rows, rank_bound, rank_bound, factor_arg, rows, tau_arg,
                              work_arg, lwork, dev_info.get()));
        check_dev_info(dev_info.get(), "orgqr/ungqr");

        cu_qr_to_row_major_kernel<<<block_count(rows64 * rank_bound64), kThreadsPerBlock>>>(
          q_ptr, factor.get(), rows64, rank_bound64, rows64, false);
        CYTNX_CHECK_CUDA_LAUNCH(cu_qr_to_row_major_kernel);
      }

    }  // namespace

    void cuQr(const Tensor &in, Tensor &Q, Tensor &R) {
      cytnx_error_msg(in.rank() != 2, "[cuQr] the input must be a rank-2 tensor, got rank %d.\n",
                      static_cast<int>(in.rank()));
      cytnx_error_msg(!in.is_contiguous(), "[cuQr] the input must be contiguous.%s", "\n");
      const std::uint64_t rows = in.shape()[0];
      const std::uint64_t cols = in.shape()[1];
      const std::uint64_t rank_bound = std::min(rows, cols);
      cytnx_error_msg(rank_bound == 0, "[cuQr] the input must not have a zero extent.%s", "\n");

      // cusolverDn<t>geqrf / orgqr / ungqr take 32-bit int dimensions and have no 64-bit variant.
      const std::uint64_t int_max = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
      cytnx_error_msg(rows > int_max || cols > int_max,
                      "[cuQr] matrix of shape (%llu, %llu) exceeds the 32-bit dimension limit "
                      "(%d) of the cuSOLVER QR routines.\n",
                      static_cast<unsigned long long>(rows), static_cast<unsigned long long>(cols),
                      std::numeric_limits<int>::max());

      cytnx_error_msg(Q.shape() != std::vector<cytnx_uint64>({rows, rank_bound}) ||
                        R.shape() != std::vector<cytnx_uint64>({rank_bound, cols}),
                      "[cuQr] Q and R must be pre-allocated with shapes (M, K) and (K, N) for an "
                      "M x N input, K = min(M, N).%s",
                      "\n");
      cytnx_error_msg(Q.device() != in.device() || R.device() != in.device(),
                      "[cuQr] Q and R must be on the same device as the input.%s", "\n");

      std::visit(
        [&](auto in_ptr) {
          using T = std::remove_pointer_t<decltype(in_ptr)>;
          if constexpr (CusolverQrScalar<T>) {
            // gpu_ptr_as<T>() verifies that each output has the dtype of the input.
            cu_qr_typed<T>(in_ptr, Q.gpu_ptr_as<T>(), R.gpu_ptr_as<T>(), static_cast<int>(rows),
                           static_cast<int>(cols));
          } else {
            cytnx_error_msg(true,
                            "[cuQr] unsupported dtype %s; expected Float, Double, ComplexFloat or "
                            "ComplexDouble.\n",
                            Type.getname(in.dtype()).c_str());
          }
        },
        in.gpu_ptr());
    }

  }  // namespace linalg_internal
}  // namespace cytnx
