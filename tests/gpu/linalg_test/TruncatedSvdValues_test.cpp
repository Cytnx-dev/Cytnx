#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

#include "Device.hpp"
#include "linalg.hpp"
#include "Tensor.hpp"
#include "Type.hpp"
#include "UniTensor.hpp"

namespace cytnx {
  namespace gpu_test {
    namespace {

      // Value tests for the truncated SVD (Gesvd_truncate, Rsvd), issue #1250.
      //
      // The input is A = H(v) * D * H(w): D holds hand-chosen singular values on its diagonal and
      // H(x) = 1 - 2 x x^dagger / |x|^2 is a Householder reflector, unitary for any nonzero x. The
      // singular values of A, the number of kept ones, the truncation error and the truncated
      // matrix H(v) * D_kept * H(w) are therefore known by construction; none of them is taken
      // from another Cytnx code path. All singular values differ, so the truncated matrix is
      // unique although U and vT are not.

      using Complex = std::complex<double>;

      const std::vector<double> kSingularValues = {8., 4., 2., 1., 0.5, 0.25};

      const std::vector<Complex> kLeftVector = {{1., 0.5},  {2., -1.},   {-1., 2.},   {0.5, 1.},
                                                {3., -0.5}, {-2., 0.25}, {1.5, -1.5}, {-0.5, 1.}};
      const std::vector<Complex> kRightVector = {{2., -1.}, {-1., 0.5},  {0.5, 1.5},    {1., -2.},
                                                 {-3., 1.}, {1.5, 0.75}, {-0.5, -0.25}, {2.5, 0.5}};

      template <typename T>
      constexpr bool is_complex_v = !std::is_floating_point_v<T>;

      template <typename T>
      using RealOf = decltype(std::abs(T{}));

      template <typename T>
      T from_complex(Complex value) {
        if constexpr (is_complex_v<T>) {
          return T(static_cast<RealOf<T>>(value.real()), static_cast<RealOf<T>>(value.imag()));
        } else {
          return static_cast<T>(value.real());
        }
      }

      // Row-major H(x) of the given size, built from the first `size` entries of `x`. The
      // imaginary parts are dropped for real dtypes, which keeps the reflector orthogonal.
      std::vector<Complex> householder(const std::vector<Complex>& x, std::size_t size,
                                       bool is_complex) {
        std::vector<Complex> v(x.begin(), x.begin() + size);
        if (!is_complex) {
          for (Complex& entry : v) entry = entry.real();
        }
        double norm2 = 0.;
        for (const Complex& entry : v) norm2 += std::norm(entry);
        std::vector<Complex> h(size * size);
        for (std::size_t i = 0; i < size; i++) {
          for (std::size_t j = 0; j < size; j++) {
            h[i * size + j] = (i == j ? 1. : 0.) - 2. * v[i] * std::conj(v[j]) / norm2;
          }
        }
        return h;
      }

      // Row-major rows x cols matrix H(v) * D * H(w), where D carries the first `kept` entries of
      // `singular_values` on its diagonal and zeros elsewhere.
      std::vector<Complex> embedded_matrix(std::size_t rows, std::size_t cols,
                                           const std::vector<double>& singular_values,
                                           std::size_t kept, bool is_complex) {
        const std::vector<Complex> left = householder(kLeftVector, rows, is_complex);
        const std::vector<Complex> right = householder(kRightVector, cols, is_complex);
        std::vector<Complex> out(rows * cols, 0.);
        for (std::size_t i = 0; i < rows; i++) {
          for (std::size_t j = 0; j < cols; j++) {
            for (std::size_t k = 0; k < kept; k++) {
              out[i * cols + j] += left[i * rows + k] * singular_values[k] * right[k * cols + j];
            }
          }
        }
        return out;
      }

      template <typename T>
      Tensor make_matrix(const std::vector<Complex>& values, std::size_t rows, std::size_t cols,
                         int device) {
        Tensor out({rows, cols}, Type.cy_typeid_v<T>);
        T* data = out.ptr_as<T>();
        for (std::size_t i = 0; i < values.size(); i++) data[i] = from_complex<T>(values[i]);
        return out.to(device);
      }

      // Copies the elements to the host. The dtype must already have been checked to be T.
      template <typename T>
      std::vector<Complex> host_values(const Tensor& tensor) {
        Tensor host = tensor.to(Device.cpu).contiguous();
        const T* data = host.ptr_as<T>();
        std::vector<Complex> out(host.storage().size());
        for (std::size_t i = 0; i < out.size(); i++) out[i] = Complex(data[i]);
        return out;
      }

      template <typename T>
      double tolerance() {
        return std::is_same_v<RealOf<T>, float> ? 2e-4 : 1e-10;
      }

      struct Cut {
        std::size_t keepdim;
        double err;
        std::size_t mindim;
        std::size_t kept;  // expected number of kept singular values of kSingularValues
      };

      const std::vector<Cut> kCuts = {
        {2, 0., 1, 2},  // cut by keepdim
        {6, 1.5, 1, 3},  // cut by err
        {4, 1.5, 1, 3},  // keepdim and err, err decides
        {3, 0.75, 1, 3},  // keepdim and err, keepdim decides
        {6, 0., 1, 6},  // nothing is cut
        {8, 0.1, 1, 6},  // keepdim above the number of singular values, nothing is cut
        {6, 1e9, 3, 3},  // err would cut everything, mindim decides
        {6, 1e9, 1, 1},  // err would cut everything, one value is always kept
      };

      // Checks `out` = [S, U, vT] or [S, U, vT, terr] against the expected values for a matrix
      // with the given singular values of which `kept` survive, the next being the first
      // discarded one.
      template <typename T>
      void check_truncated_svd(std::vector<Tensor> out, std::size_t rows, std::size_t cols,
                               const std::vector<double>& singular_values, std::size_t kept,
                               unsigned int return_err, int device,
                               const std::vector<Complex>& expected_truncated_matrix,
                               double discarded_tolerance) {
        const unsigned int matrix_dtype = Type.cy_typeid_v<T>;
        const unsigned int real_dtype = Type.cy_typeid_v<RealOf<T>>;
        const double tol = tolerance<T>();

        ASSERT_EQ(out.size(), return_err ? 4u : 3u);
        for (const Tensor& tensor : out) EXPECT_EQ(tensor.device(), device);

        Tensor& s = out[0];
        Tensor& u = out[1];
        Tensor& vt = out[2];
        ASSERT_EQ(s.dtype(), real_dtype);
        ASSERT_EQ(u.dtype(), matrix_dtype);
        ASSERT_EQ(vt.dtype(), matrix_dtype);
        ASSERT_EQ(s.shape(), std::vector<cytnx_uint64>({kept}));
        ASSERT_EQ(u.shape(), std::vector<cytnx_uint64>({rows, kept}));
        ASSERT_EQ(vt.shape(), std::vector<cytnx_uint64>({kept, cols}));

        const std::vector<Complex> s_host = host_values<RealOf<T>>(s);
        for (std::size_t k = 0; k < kept; k++) {
          EXPECT_NEAR(s_host[k].real(), singular_values[k], tol) << "S[" << k << "]";
        }

        const std::vector<Complex> u_host = host_values<T>(u);
        const std::vector<Complex> vt_host = host_values<T>(vt);
        double max_residual = 0.;
        for (std::size_t i = 0; i < rows; i++) {
          for (std::size_t j = 0; j < cols; j++) {
            Complex element = 0.;
            for (std::size_t k = 0; k < kept; k++) {
              element += u_host[i * kept + k] * s_host[k] * vt_host[k * cols + j];
            }
            max_residual =
              std::max(max_residual, std::abs(element - expected_truncated_matrix[i * cols + j]));
          }
        }
        EXPECT_LT(max_residual, 10. * tol) << "U diag(S) vT differs from the truncated matrix";

        if (!return_err) return;
        Tensor& terr = out[3];
        ASSERT_EQ(terr.dtype(), real_dtype);
        const std::size_t discarded = singular_values.size() - kept;
        if (return_err == 1) {
          ASSERT_EQ(terr.rank(), 0u);
          const std::vector<Complex> terr_host = host_values<RealOf<T>>(terr);
          ASSERT_EQ(terr_host.size(), 1u);
          if (discarded == 0) {
            EXPECT_EQ(terr_host[0].real(), 0.);
          } else {
            EXPECT_NEAR(terr_host[0].real(), singular_values[kept], discarded_tolerance);
          }
        } else {
          ASSERT_EQ(terr.shape(), std::vector<cytnx_uint64>({discarded}));
          const std::vector<Complex> terr_host = host_values<RealOf<T>>(terr);
          ASSERT_EQ(terr_host.size(), discarded);
          for (std::size_t k = 0; k < discarded; k++) {
            EXPECT_NEAR(terr_host[k].real(), singular_values[kept + k], discarded_tolerance)
              << "terr[" << k << "]";
          }
        }
      }

      // The blocks of [S, U, vT, terr?] as returned for a dense UniTensor.
      std::vector<Tensor> blocks_of(const std::vector<UniTensor>& unitensors) {
        std::vector<Tensor> out;
        for (const UniTensor& unitensor : unitensors) out.push_back(unitensor.get_block_());
        return out;
      }

      enum class Function { kGesvdTruncate, kRsvd };
      enum class Container { kTensor, kDenseUniTensor };

      // Oversampling that makes Rsvd take as many samples as there are singular values, so it runs
      // a full SVD and its results are exact.
      const cytnx_uint64 kFullOversampling = 10;

      std::vector<Tensor> run(Function function, Container container, const Tensor& matrix,
                              const Cut& cut, unsigned int return_err) {
        if (container == Container::kTensor) {
          if (function == Function::kGesvdTruncate) {
            return linalg::Gesvd_truncate(matrix, cut.keepdim, cut.err, true, true, return_err,
                                          cut.mindim);
          }
          return linalg::Rsvd(matrix, cut.keepdim, cut.err, true, true, return_err, cut.mindim,
                              kFullOversampling);
        }
        const UniTensor unitensor(matrix, false, 1);
        if (function == Function::kGesvdTruncate) {
          return blocks_of(linalg::Gesvd_truncate(unitensor, cut.keepdim, cut.err, true, true,
                                                  return_err, cut.mindim));
        }
        return blocks_of(linalg::Rsvd(unitensor, cut.keepdim, cut.err, true, true, return_err,
                                      cut.mindim, kFullOversampling));
      }

      // The host results are checked against the same expectations, which shows that the expected
      // values are the ones the CPU implementation returns.
      template <typename T>
      void check_all_cuts(Function function, Container container) {
        const std::vector<std::vector<std::size_t>> shapes = {{6, 8}, {8, 6}, {6, 6}};
        for (int device : {Device.cpu, Device.cuda}) {
          for (const std::vector<std::size_t>& shape : shapes) {
            const std::size_t rows = shape[0], cols = shape[1];
            const Tensor matrix = make_matrix<T>(
              embedded_matrix(rows, cols, kSingularValues, kSingularValues.size(), is_complex_v<T>),
              rows, cols, device);
            for (const Cut& cut : kCuts) {
              const std::vector<Complex> expected =
                embedded_matrix(rows, cols, kSingularValues, cut.kept, is_complex_v<T>);
              for (unsigned int return_err : {0u, 1u, 2u}) {
                SCOPED_TRACE(std::string(device == Device.cpu ? "cpu" : "cuda") + ", shape " +
                             std::to_string(rows) + "x" + std::to_string(cols) + ", keepdim " +
                             std::to_string(cut.keepdim) + ", err " + std::to_string(cut.err) +
                             ", mindim " + std::to_string(cut.mindim) + ", return_err " +
                             std::to_string(return_err));
                check_truncated_svd<T>(run(function, container, matrix, cut, return_err), rows,
                                       cols, kSingularValues, cut.kept, return_err, device,
                                       expected, tolerance<T>());
              }
            }
          }
        }
      }

      TEST(GesvdTruncateValues, GpuTensorFloat) {
        check_all_cuts<cytnx_float>(Function::kGesvdTruncate, Container::kTensor);
      }
      TEST(GesvdTruncateValues, GpuTensorDouble) {
        check_all_cuts<cytnx_double>(Function::kGesvdTruncate, Container::kTensor);
      }
      TEST(GesvdTruncateValues, GpuTensorComplexFloat) {
        check_all_cuts<cytnx_complex64>(Function::kGesvdTruncate, Container::kTensor);
      }
      TEST(GesvdTruncateValues, GpuTensorComplexDouble) {
        check_all_cuts<cytnx_complex128>(Function::kGesvdTruncate, Container::kTensor);
      }

      TEST(GesvdTruncateValues, GpuDenseUniTensorFloat) {
        check_all_cuts<cytnx_float>(Function::kGesvdTruncate, Container::kDenseUniTensor);
      }
      TEST(GesvdTruncateValues, GpuDenseUniTensorDouble) {
        check_all_cuts<cytnx_double>(Function::kGesvdTruncate, Container::kDenseUniTensor);
      }
      TEST(GesvdTruncateValues, GpuDenseUniTensorComplexFloat) {
        check_all_cuts<cytnx_complex64>(Function::kGesvdTruncate, Container::kDenseUniTensor);
      }
      TEST(GesvdTruncateValues, GpuDenseUniTensorComplexDouble) {
        check_all_cuts<cytnx_complex128>(Function::kGesvdTruncate, Container::kDenseUniTensor);
      }

      TEST(RsvdValues, GpuTensorFloat) {
        check_all_cuts<cytnx_float>(Function::kRsvd, Container::kTensor);
      }
      TEST(RsvdValues, GpuTensorDouble) {
        check_all_cuts<cytnx_double>(Function::kRsvd, Container::kTensor);
      }
      TEST(RsvdValues, GpuTensorComplexFloat) {
        check_all_cuts<cytnx_complex64>(Function::kRsvd, Container::kTensor);
      }
      TEST(RsvdValues, GpuTensorComplexDouble) {
        check_all_cuts<cytnx_complex128>(Function::kRsvd, Container::kTensor);
      }

      TEST(RsvdValues, GpuDenseUniTensorFloat) {
        check_all_cuts<cytnx_float>(Function::kRsvd, Container::kDenseUniTensor);
      }
      TEST(RsvdValues, GpuDenseUniTensorDouble) {
        check_all_cuts<cytnx_double>(Function::kRsvd, Container::kDenseUniTensor);
      }
      TEST(RsvdValues, GpuDenseUniTensorComplexFloat) {
        check_all_cuts<cytnx_complex64>(Function::kRsvd, Container::kDenseUniTensor);
      }
      TEST(RsvdValues, GpuDenseUniTensorComplexDouble) {
        check_all_cuts<cytnx_complex128>(Function::kRsvd, Container::kDenseUniTensor);
      }

      // Rsvd with a random projection: 4 samples of a rank-3 matrix capture its range exactly, so
      // the three nonzero singular values and the matrix itself are recovered up to rounding. The
      // SVD of the projected matrix has 4 singular values, of which the discarded fourth is zero
      // up to rounding.
      template <typename T>
      void check_projected_rsvd() {
        const std::vector<double> projected_singular_values = {8., 4., 2., 0.};
        const std::size_t kept = 3;
        const std::vector<std::vector<std::size_t>> shapes = {{6, 8}, {8, 6}};
        for (const std::vector<std::size_t>& shape : shapes) {
          const std::size_t rows = shape[0], cols = shape[1];
          const std::vector<Complex> values =
            embedded_matrix(rows, cols, projected_singular_values, kept, is_complex_v<T>);
          const Tensor matrix = make_matrix<T>(values, rows, cols, Device.cuda);
          for (unsigned int return_err : {0u, 1u, 2u}) {
            SCOPED_TRACE("shape " + std::to_string(rows) + "x" + std::to_string(cols) +
                         ", return_err " + std::to_string(return_err));
            const std::vector<Tensor> out =
              linalg::Rsvd(matrix, kept, 0., true, true, return_err, 1, /*oversampling_summand=*/1,
                           /*oversampling_factor=*/0., /*power_iteration=*/2, /*seed=*/7);
            check_truncated_svd<T>(out, rows, cols, projected_singular_values, kept, return_err,
                                   Device.cuda, values, 100. * tolerance<T>());
          }
        }
      }

      TEST(RsvdValues, GpuProjectedLowRank) {
#ifndef UNI_CUQUANTUM
        GTEST_SKIP() << "GPU randomized SVD dense tests require cuQuantum-enabled GPU QR support.";
#endif
        {
          SCOPED_TRACE("Float");
          check_projected_rsvd<cytnx_float>();
        }
        {
          SCOPED_TRACE("Double");
          check_projected_rsvd<cytnx_double>();
        }
        {
          SCOPED_TRACE("ComplexFloat");
          check_projected_rsvd<cytnx_complex64>();
        }
        {
          SCOPED_TRACE("ComplexDouble");
          check_projected_rsvd<cytnx_complex128>();
        }
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
