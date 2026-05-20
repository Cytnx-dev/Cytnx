#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_GEMM_BATCH_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_GEMM_BATCH_INTERNAL_H_

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <vector>

#ifdef UNI_MKL
  #include <mkl.h>
#endif

#include "Type.hpp"
#include "backend/Scalar.hpp"
#include "backend/lapack_wrapper.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace linalg_internal {

    namespace detail {

#ifdef UNI_MKL
      template <class T>
      struct MKLTraits;
      template <>
      struct MKLTraits<cytnx_complex128> {
        static constexpr auto gemm = zgemm_batch;
      };
      template <>
      struct MKLTraits<cytnx_complex64> {
        static constexpr auto gemm = cgemm_batch;
      };
      template <>
      struct MKLTraits<cytnx_double> {
        static constexpr auto gemm = dgemm_batch;
      };
      template <>
      struct MKLTraits<cytnx_float> {
        static constexpr auto gemm = sgemm_batch;
      };
#endif

      template <class T>
      T ScalarTo(const Scalar& x) {
        if constexpr (is_complex_v<T>) {
          if constexpr (std::is_same_v<T, cytnx_complex128>)
            return x._impl->to_cytnx_complex128();
          else
            return x._impl->to_cytnx_complex64();
        } else {
          return static_cast<T>(x);
        }
      }

      template <class T>
      std::vector<T> ToMkl(const std::vector<Scalar>& xs) {
        std::vector<T> ys;
        ys.reserve(xs.size());
        std::transform(xs.begin(), xs.end(), std::back_inserter(ys), ScalarTo<T>);
        return ys;
      }

    }  // namespace detail

    template <class T>
    void Gemm_Batch(const char* transa, const char* transb, const blas_int* m, const blas_int* n,
                    const blas_int* k, const std::vector<Scalar>& alpha, const void** a,
                    const blas_int* lda, const void** b, const blas_int* ldb,
                    const std::vector<Scalar>& beta, void** c, const blas_int* ldc,
                    blas_int group_count, const blas_int* group_size) {
#ifdef UNI_MKL
      const auto alphas = detail::ToMkl<T>(alpha);
      const auto betas = detail::ToMkl<T>(beta);
      detail::MKLTraits<T>::gemm(transa, transb, m, n, k, alphas.data(),
                                 reinterpret_cast<const T**>(a), lda,
                                 reinterpret_cast<const T**>(b), ldb, betas.data(),
                                 reinterpret_cast<T**>(c), ldc, &group_count, group_size);
#else
      cytnx_error_msg(true, "[Gemm_Batch_internal] fatal error, Gemm_Batch required MKL.%s", "\n");
#endif
    }

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_GEMM_BATCH_INTERNAL_H_
