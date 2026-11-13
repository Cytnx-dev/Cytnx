#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "cytnx.hpp"

// GPU QR tests. Expected values come from host code in this file (a textbook Householder QR in
// the LAPACK geqrf convention, plain matrix products, and hand-computed cases), never from another
// Cytnx linear-algebra path.

namespace cytnx {
  namespace gpu_test {
    namespace {

      using Complex = std::complex<double>;

      // Row-major host matrix in double-precision complex arithmetic.
      struct HostMatrix {
        std::size_t rows = 0;
        std::size_t cols = 0;
        std::vector<Complex> data;

        HostMatrix(std::size_t rows_in, std::size_t cols_in)
            : rows(rows_in), cols(cols_in), data(rows_in * cols_in, Complex(0.0, 0.0)) {}

        Complex& operator()(std::size_t i, std::size_t j) { return data[i * cols + j]; }
        Complex operator()(std::size_t i, std::size_t j) const { return data[i * cols + j]; }
      };

      struct HostQr {
        HostMatrix q;
        HostMatrix r;
      };

      bool is_complex_dtype(unsigned int dtype) {
        return dtype == Type.ComplexDouble || dtype == Type.ComplexFloat;
      }

      bool is_single_precision(unsigned int dtype) {
        return dtype == Type.ComplexFloat || dtype == Type.Float;
      }

      // Fractional values of both signs in (-1, 1); complex dtypes get a nonzero imaginary part.
      HostMatrix random_matrix(std::size_t rows, std::size_t cols, bool is_complex,
                               unsigned int seed) {
        std::mt19937_64 engine(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        HostMatrix out(rows, cols);
        for (Complex& value : out.data) {
          const double re = dist(engine);
          const double im = is_complex ? dist(engine) : 0.0;
          value = Complex(re, im);
        }
        return out;
      }

      HostMatrix multiply(const HostMatrix& lhs, const HostMatrix& rhs) {
        HostMatrix out(lhs.rows, rhs.cols);
        for (std::size_t i = 0; i < lhs.rows; ++i)
          for (std::size_t k = 0; k < lhs.cols; ++k)
            for (std::size_t j = 0; j < rhs.cols; ++j) out(i, j) += lhs(i, k) * rhs(k, j);
        return out;
      }

      HostMatrix conjugate_transpose(const HostMatrix& in) {
        HostMatrix out(in.cols, in.rows);
        for (std::size_t i = 0; i < in.rows; ++i)
          for (std::size_t j = 0; j < in.cols; ++j) out(j, i) = std::conj(in(i, j));
        return out;
      }

      HostMatrix identity(std::size_t n) {
        HostMatrix out(n, n);
        for (std::size_t i = 0; i < n; ++i) out(i, i) = 1.0;
        return out;
      }

      double max_abs(const HostMatrix& in) {
        double out = 0.0;
        for (Complex value : in.data) out = std::max(out, std::abs(value));
        return out;
      }

      double max_abs_diff(const HostMatrix& lhs, const HostMatrix& rhs) {
        EXPECT_EQ(lhs.rows, rhs.rows);
        EXPECT_EQ(lhs.cols, rhs.cols);
        if (lhs.rows != rhs.rows || lhs.cols != rhs.cols) return INFINITY;
        double out = 0.0;
        for (std::size_t i = 0; i < lhs.data.size(); ++i)
          out = std::max(out, std::abs(lhs.data[i] - rhs.data[i]));
        return out;
      }

      Tensor to_tensor(const HostMatrix& in, unsigned int dtype) {
        Tensor out({in.rows, in.cols}, dtype, Device.cpu);
        for (cytnx_uint64 i = 0; i < in.rows; ++i) {
          for (cytnx_uint64 j = 0; j < in.cols; ++j) {
            const Complex value = in(i, j);
            if (dtype == Type.ComplexDouble) {
              out.at<cytnx_complex128>({i, j}) = value;
            } else if (dtype == Type.ComplexFloat) {
              out.at<cytnx_complex64>({i, j}) =
                cytnx_complex64(static_cast<float>(value.real()), static_cast<float>(value.imag()));
            } else if (dtype == Type.Double) {
              out.at<cytnx_double>({i, j}) = value.real();
            } else if (dtype == Type.Float) {
              out.at<cytnx_float>({i, j}) = static_cast<float>(value.real());
            } else if (dtype == Type.Int64) {
              out.at<cytnx_int64>({i, j}) = static_cast<cytnx_int64>(value.real());
            } else {
              ADD_FAILURE() << "unsupported test dtype " << dtype;
            }
          }
        }
        return out;
      }

      Complex host_element(const Tensor& host, const std::vector<cytnx_uint64>& index) {
        const unsigned int dtype = host.dtype();
        if (dtype == Type.ComplexDouble) return host.at<cytnx_complex128>(index);
        if (dtype == Type.ComplexFloat) {
          const cytnx_complex64 value = host.at<cytnx_complex64>(index);
          return Complex(value.real(), value.imag());
        }
        if (dtype == Type.Double) return Complex(host.at<cytnx_double>(index), 0.0);
        if (dtype == Type.Float) return Complex(host.at<cytnx_float>(index), 0.0);
        if (dtype == Type.Int64)
          return Complex(static_cast<double>(host.at<cytnx_int64>(index)), 0.0);
        ADD_FAILURE() << "unsupported test dtype " << dtype;
        return Complex(0.0, 0.0);
      }

      // Copies a rank-2 tensor (any device) into a host matrix.
      HostMatrix to_host(const Tensor& in) {
        const Tensor host = in.to(Device.cpu).contiguous();
        EXPECT_EQ(host.shape().size(), 2u);
        if (host.shape().size() != 2) return HostMatrix(0, 0);
        HostMatrix out(host.shape()[0], host.shape()[1]);
        for (cytnx_uint64 i = 0; i < out.rows; ++i)
          for (cytnx_uint64 j = 0; j < out.cols; ++j) out(i, j) = host_element(host, {i, j});
        return out;
      }

      // Unblocked Householder QR in the LAPACK geqrf / geqr2 / larfg convention:
      //   A = Q R,  Q = H_1 ... H_K,  H_i = I - tau_i v_i v_i^H,  v_i[i] = 1,
      // with beta_i = R(i, i) real and of the sign opposite to the real part of the pivot.
      HostQr reference_householder_qr(const HostMatrix& a) {
        const std::size_t rows = a.rows;
        const std::size_t cols = a.cols;
        const std::size_t rank_bound = std::min(rows, cols);
        HostMatrix work = a;
        std::vector<std::vector<Complex>> reflectors(rank_bound);
        std::vector<Complex> tau(rank_bound, Complex(0.0, 0.0));

        for (std::size_t k = 0; k < rank_bound; ++k) {
          const Complex alpha = work(k, k);
          double tail_norm_sq = 0.0;
          for (std::size_t i = k + 1; i < rows; ++i) tail_norm_sq += std::norm(work(i, k));

          std::vector<Complex> v(rows, Complex(0.0, 0.0));
          v[k] = 1.0;
          if (tail_norm_sq == 0.0 && alpha.imag() == 0.0) {
            tau[k] = 0.0;  // H_k is the identity
          } else {
            const double norm = std::sqrt(std::norm(alpha) + tail_norm_sq);
            const double beta = alpha.real() >= 0.0 ? -norm : norm;
            tau[k] = Complex((beta - alpha.real()) / beta, -alpha.imag() / beta);
            for (std::size_t i = k + 1; i < rows; ++i) v[i] = work(i, k) / (alpha - beta);
            // work <- H_k^H work = (I - conj(tau) v v^H) work on the trailing columns
            for (std::size_t j = k; j < cols; ++j) {
              Complex dot(0.0, 0.0);
              for (std::size_t i = k; i < rows; ++i) dot += std::conj(v[i]) * work(i, j);
              for (std::size_t i = k; i < rows; ++i) work(i, j) -= std::conj(tau[k]) * v[i] * dot;
            }
            work(k, k) = beta;
            for (std::size_t i = k + 1; i < rows; ++i) work(i, k) = 0.0;
          }
          reflectors[k] = v;
        }

        HostQr out{HostMatrix(rows, rank_bound), HostMatrix(rank_bound, cols)};
        for (std::size_t i = 0; i < rank_bound; ++i)
          for (std::size_t j = i; j < cols; ++j) out.r(i, j) = work(i, j);

        // Q = H_1 ... H_K applied to the first K columns of the identity.
        for (std::size_t i = 0; i < rank_bound; ++i) out.q(i, i) = 1.0;
        for (std::size_t k = rank_bound; k-- > 0;) {
          const std::vector<Complex>& v = reflectors[k];
          for (std::size_t j = 0; j < rank_bound; ++j) {
            Complex dot(0.0, 0.0);
            for (std::size_t i = k; i < rows; ++i) dot += std::conj(v[i]) * out.q(i, j);
            for (std::size_t i = k; i < rows; ++i) out.q(i, j) -= tau[k] * v[i] * dot;
          }
        }
        return out;
      }

      // Among the valid QR factorizations, the cuSOLVER path returns the LAPACK Householder one
      // (R with a real diagonal of the sign opposite to the pivot). cuQuantum builds route Qr
      // through cutensornetTensorQR, which does not document its choice, so comparisons with
      // specific Q and R values are limited to builds without cuQuantum.
#ifdef UNI_CUQUANTUM
      constexpr bool kHouseholderConvention = false;
#else
      constexpr bool kHouseholderConvention = true;
#endif

      double tolerance(unsigned int dtype) { return is_single_precision(dtype) ? 2e-4 : 1e-11; }

      void expect_output_layout(const std::vector<Tensor>& out,
                                const std::vector<cytnx_uint64>& shape, unsigned int dtype) {
        const cytnx_uint64 rank_bound = std::min(shape[0], shape[1]);
        ASSERT_EQ(out.size(), 2u);
        EXPECT_EQ(out[0].shape(), std::vector<cytnx_uint64>({shape[0], rank_bound}));
        EXPECT_EQ(out[1].shape(), std::vector<cytnx_uint64>({rank_bound, shape[1]}));
        for (const Tensor& tensor : out) {
          EXPECT_EQ(tensor.dtype(), dtype);
          EXPECT_EQ(tensor.device(), Device.cuda);
        }
      }

      // Q^H Q = I, Q R = A, and R exactly zero below the diagonal.
      void expect_valid_factorization(const HostMatrix& a, const HostMatrix& q, const HostMatrix& r,
                                      double tol) {
        const double scale = std::max(1.0, max_abs(a)) * static_cast<double>(a.rows + a.cols);
        EXPECT_LE(max_abs_diff(multiply(conjugate_transpose(q), q), identity(q.cols)), tol * scale)
          << "Q does not have orthonormal columns";
        EXPECT_LE(max_abs_diff(multiply(q, r), a), tol * scale) << "Q R != A";
        for (std::size_t i = 0; i < r.rows; ++i)
          for (std::size_t j = 0; j < std::min(i, r.cols); ++j)
            EXPECT_EQ(r(i, j), Complex(0.0, 0.0)) << "R(" << i << "," << j << ") is not zero";
      }

      // Full check for a matrix of full rank, where the Householder factorization is well defined
      // to rounding: invariants plus Q and R against the host reference.
      void check_full_rank_case(const HostMatrix& requested, unsigned int dtype) {
        SCOPED_TRACE("dtype " + Type.getname(dtype) + ", shape " + std::to_string(requested.rows) +
                     "x" + std::to_string(requested.cols));
        const Tensor cpu_input = to_tensor(requested, dtype);
        const HostMatrix a = to_host(cpu_input);  // exactly the stored (possibly float) values
        const double tol = tolerance(dtype);
        const double scale = std::max(1.0, max_abs(a)) * static_cast<double>(a.rows + a.cols);

        const std::vector<Tensor> out = linalg::Qr(cpu_input.to(Device.cuda));
        expect_output_layout(out, {a.rows, a.cols}, dtype);
        const HostMatrix q = to_host(out[0]);
        const HostMatrix r = to_host(out[1]);
        expect_valid_factorization(a, q, r, tol);

        if (kHouseholderConvention) {
          const HostQr expected = reference_householder_qr(a);
          EXPECT_LE(max_abs_diff(q, expected.q), tol * scale) << "Q differs from the reference";
          EXPECT_LE(max_abs_diff(r, expected.r), tol * scale) << "R differs from the reference";
        }
      }

      const std::vector<unsigned int> kFloatingDtypes = {Type.ComplexDouble, Type.ComplexFloat,
                                                         Type.Double, Type.Float};

      void check_shape_for_all_dtypes(std::size_t rows, std::size_t cols, unsigned int seed) {
        for (unsigned int dtype : kFloatingDtypes) {
          check_full_rank_case(random_matrix(rows, cols, is_complex_dtype(dtype), seed), dtype);
        }
      }

      TEST(Qr, GpuTallMatchesHouseholderReference) {
        check_shape_for_all_dtypes(7, 4, 11);
        check_shape_for_all_dtypes(5, 1, 12);
        check_shape_for_all_dtypes(96, 64, 13);
      }

      TEST(Qr, GpuSquareMatchesHouseholderReference) {
        check_shape_for_all_dtypes(1, 1, 21);
        check_shape_for_all_dtypes(5, 5, 22);
        check_shape_for_all_dtypes(80, 80, 23);
      }

      TEST(Qr, GpuWideMatchesHouseholderReference) {
        check_shape_for_all_dtypes(4, 7, 31);
        check_shape_for_all_dtypes(1, 5, 32);
        check_shape_for_all_dtypes(64, 96, 33);
      }

      // A = [3; 4]: beta = -5, v = [1, 1/2], tau = (beta - 3) / beta = 8/5, so Q = [-3/5; -4/5]
      // and R = [-5].
      TEST(Qr, GpuHandComputedRealColumn) {
        if (!kHouseholderConvention) GTEST_SKIP() << "the sign of Q and R is cuSOLVER-specific";
        for (unsigned int dtype : {Type.Double, Type.Float}) {
          SCOPED_TRACE("dtype " + Type.getname(dtype));
          HostMatrix a(2, 1);
          a(0, 0) = 3.0;
          a(1, 0) = 4.0;
          const std::vector<Tensor> out = linalg::Qr(to_tensor(a, dtype).to(Device.cuda));
          expect_output_layout(out, {2, 1}, dtype);
          const HostMatrix q = to_host(out[0]);
          const HostMatrix r = to_host(out[1]);
          const double tol = tolerance(dtype);
          EXPECT_NEAR(q(0, 0).real(), -0.6, tol);
          EXPECT_NEAR(q(1, 0).real(), -0.8, tol);
          EXPECT_NEAR(r(0, 0).real(), -5.0, 5 * tol);
        }
      }

      // A = [3i; 4]: alpha = 3i has zero real part, so beta = -5 and the geqrf scalar is
      // ((beta - 0) / beta, -3 / beta) = 1 + 0.6i, v = [1, 4 / (3i + 5)] = [1, (10 - 6i) / 17].
      // Q = H e_1 = e_1 - tau v = [-0.6i; -0.8], R = [-5], and Q R = [3i; 4].
      TEST(Qr, GpuHandComputedComplexColumn) {
        if (!kHouseholderConvention) GTEST_SKIP() << "the phase of Q and R is cuSOLVER-specific";
        for (unsigned int dtype : {Type.ComplexDouble, Type.ComplexFloat}) {
          SCOPED_TRACE("dtype " + Type.getname(dtype));
          HostMatrix a(2, 1);
          a(0, 0) = Complex(0.0, 3.0);
          a(1, 0) = 4.0;
          const std::vector<Tensor> out = linalg::Qr(to_tensor(a, dtype).to(Device.cuda));
          expect_output_layout(out, {2, 1}, dtype);
          const HostMatrix q = to_host(out[0]);
          const HostMatrix r = to_host(out[1]);
          const double tol = tolerance(dtype);
          EXPECT_LE(std::abs(q(0, 0) - Complex(0.0, -0.6)), tol);
          EXPECT_LE(std::abs(q(1, 0) - Complex(-0.8, 0.0)), tol);
          EXPECT_LE(std::abs(r(0, 0) - Complex(-5.0, 0.0)), 5 * tol);
        }
      }

      // Rank-deficient input: the factorization is still exact, but which orthonormal completion
      // Q gets depends on rounding, so only the invariants are checked.
      TEST(Qr, GpuRankDeficient) {
        struct Shape {
          std::size_t rows;
          std::size_t cols;
        };
        for (unsigned int dtype : kFloatingDtypes) {
          for (const Shape shape : {Shape{6, 5}, Shape{5, 5}, Shape{4, 6}}) {
            SCOPED_TRACE("dtype " + Type.getname(dtype) + ", shape " + std::to_string(shape.rows) +
                         "x" + std::to_string(shape.cols));
            const bool is_complex = is_complex_dtype(dtype);
            // rank 2 product, with one column additionally set to exactly zero
            HostMatrix requested = multiply(random_matrix(shape.rows, 2, is_complex, 41),
                                            random_matrix(2, shape.cols, is_complex, 42));
            for (std::size_t i = 0; i < shape.rows; ++i) requested(i, 1) = 0.0;
            const Tensor cpu_input = to_tensor(requested, dtype);
            const HostMatrix a = to_host(cpu_input);

            const std::vector<Tensor> out = linalg::Qr(cpu_input.to(Device.cuda));
            expect_output_layout(out, {a.rows, a.cols}, dtype);
            expect_valid_factorization(a, to_host(out[0]), to_host(out[1]), tolerance(dtype));
          }
        }
      }

      // The zero matrix needs no reflection at all: R is zero and Q still has orthonormal
      // columns. With the Householder convention Q is exactly the leading columns of the identity.
      TEST(Qr, GpuZeroMatrix) {
        for (unsigned int dtype : kFloatingDtypes) {
          SCOPED_TRACE("dtype " + Type.getname(dtype));
          const HostMatrix a(4, 3);
          const std::vector<Tensor> out = linalg::Qr(to_tensor(a, dtype).to(Device.cuda));
          expect_output_layout(out, {4, 3}, dtype);
          const HostMatrix q = to_host(out[0]);
          expect_valid_factorization(a, q, to_host(out[1]), tolerance(dtype));
          EXPECT_EQ(max_abs(to_host(out[1])), 0.0);
          if (kHouseholderConvention) {
            HostMatrix expected_q(4, 3);
            for (std::size_t i = 0; i < 3; ++i) expected_q(i, i) = 1.0;
            EXPECT_EQ(max_abs_diff(q, expected_q), 0.0);
          }
        }
      }

      // The input is made contiguous inside Qr; the factors must be those of the permuted matrix.
      TEST(Qr, GpuNonContiguousInput) {
        const HostMatrix stored = random_matrix(4, 6, true, 51);
        const Tensor permuted = to_tensor(stored, Type.ComplexDouble).to(Device.cuda).permute(1, 0);
        ASSERT_FALSE(permuted.is_contiguous());
        HostMatrix a(6, 4);
        for (std::size_t i = 0; i < 6; ++i)
          for (std::size_t j = 0; j < 4; ++j) a(i, j) = stored(j, i);

        const std::vector<Tensor> out = linalg::Qr(permuted);
        expect_output_layout(out, {6, 4}, Type.ComplexDouble);
        const double tol = 1e-11;
        expect_valid_factorization(a, to_host(out[0]), to_host(out[1]), tol);
        if (kHouseholderConvention) {
          const HostQr expected = reference_householder_qr(a);
          EXPECT_LE(max_abs_diff(to_host(out[0]), expected.q), tol);
          EXPECT_LE(max_abs_diff(to_host(out[1]), expected.r), tol);
        }
      }

      // Integer input is factored in double precision, as on the CPU.
      TEST(Qr, GpuIntegerInputPromotesToDouble) {
        HostMatrix a(3, 2);
        const double values[3][2] = {{2, -1}, {-3, 4}, {6, 5}};
        for (std::size_t i = 0; i < 3; ++i)
          for (std::size_t j = 0; j < 2; ++j) a(i, j) = values[i][j];

        const std::vector<Tensor> out = linalg::Qr(to_tensor(a, Type.Int64).to(Device.cuda));
        expect_output_layout(out, {3, 2}, Type.Double);
        const double tol = 1e-12;
        expect_valid_factorization(a, to_host(out[0]), to_host(out[1]), tol);
        // first column [2, -3, 6] has norm 7, so |R(0, 0)| = 7 in every convention
        EXPECT_NEAR(std::abs(to_host(out[1])(0, 0)), 7.0, tol);
        if (kHouseholderConvention) {
          const HostQr expected = reference_householder_qr(a);
          EXPECT_LE(max_abs_diff(to_host(out[0]), expected.q), tol);
          EXPECT_LE(max_abs_diff(to_host(out[1]), expected.r), tol);
          EXPECT_NEAR(to_host(out[1])(0, 0).real(), -7.0, tol);
        }
      }

      // Dense UniTensor: the first rowrank legs are the rows. Checks labels, shapes and, through
      // the blocks reshaped to matrices, Q^H Q = I and Q R = A.
      void check_dense_unitensor(cytnx_uint64 rowrank) {
        SCOPED_TRACE("rowrank " + std::to_string(rowrank));
        const std::vector<cytnx_uint64> shape = {3, 4, 5};
        cytnx_uint64 rows = 1;
        for (cytnx_uint64 i = 0; i < rowrank; ++i) rows *= shape[i];
        const cytnx_uint64 cols = 60 / rows;
        const cytnx_uint64 rank_bound = std::min(rows, cols);

        const HostMatrix a = random_matrix(rows, cols, true, 61);
        Tensor block = to_tensor(a, Type.ComplexDouble).reshape({3, 4, 5});
        UniTensor input = UniTensor(block, false, rowrank).relabel({"a", "b", "c"});
        const std::vector<UniTensor> out = linalg::Qr(input.to(Device.cuda));
        ASSERT_EQ(out.size(), 2u);

        const UniTensor& q_ut = out[0];
        const UniTensor& r_ut = out[1];
        std::vector<std::string> q_labels(input.labels().begin(), input.labels().begin() + rowrank);
        q_labels.push_back("_aux_");
        std::vector<std::string> r_labels = {"_aux_"};
        r_labels.insert(r_labels.end(), input.labels().begin() + rowrank, input.labels().end());
        EXPECT_EQ(q_ut.labels(), q_labels);
        EXPECT_EQ(r_ut.labels(), r_labels);
        EXPECT_EQ(q_ut.rowrank(), rowrank);
        EXPECT_EQ(r_ut.rowrank(), 1u);
        EXPECT_EQ(q_ut.device(), Device.cuda);
        EXPECT_EQ(r_ut.device(), Device.cuda);
        EXPECT_EQ(q_ut.dtype(), Type.ComplexDouble);
        EXPECT_EQ(r_ut.dtype(), Type.ComplexDouble);

        const HostMatrix q =
          to_host(q_ut.get_block().reshape(std::vector<cytnx_uint64>{rows, rank_bound}));
        const HostMatrix r =
          to_host(r_ut.get_block().reshape(std::vector<cytnx_uint64>{rank_bound, cols}));
        expect_valid_factorization(a, q, r, 1e-11);

        if (kHouseholderConvention) {
          const HostQr expected = reference_householder_qr(a);
          EXPECT_LE(max_abs_diff(q, expected.q), 1e-11);
          EXPECT_LE(max_abs_diff(r, expected.r), 1e-11);
        }
      }

      TEST(Qr, GpuDenseUniTensor) {
        check_dense_unitensor(2);  // 12 x 5, tall
        check_dense_unitensor(1);  // 3 x 20, wide
      }

      // The scalar factors of the reflectors are only returned for CPU tensors.
      TEST(Qr, GpuReturningTauIsRejected) {
        const Tensor matrix =
          to_tensor(random_matrix(4, 3, false, 71), Type.Double).to(Device.cuda);
        EXPECT_THROW(linalg::Qr(matrix, true), std::logic_error);
        const UniTensor dense = UniTensor(matrix, false, 1);
        EXPECT_THROW(linalg::Qr(dense, true), std::logic_error);
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
