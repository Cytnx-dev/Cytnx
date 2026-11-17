#ifndef CYTNX_TESTS_LINALG_TEST_EIGH_TEST_TOOLS_H_
#define CYTNX_TESTS_LINALG_TEST_EIGH_TEST_TOOLS_H_

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "cytnx.hpp"

// Hermitian matrices with spectra known in closed form, and checks of an eigendecomposition
// that use nothing of Cytnx but element access: the arithmetic is done here, on host doubles.
// Shared by the CPU and the GPU tests of linalg::Eigh.
namespace cytnx {
  namespace test {
    namespace eigh {

      // A dense n x n matrix, row-major.
      struct Matrix {
        std::size_t n = 0;
        std::vector<std::complex<double>> data;

        explicit Matrix(std::size_t size) : n(size), data(size * size) {}
        std::complex<double> &operator()(std::size_t i, std::size_t j) { return data[i * n + j]; }
        std::complex<double> operator()(std::size_t i, std::size_t j) const {
          return data[i * n + j];
        }
      };

      inline bool is_complex_dtype(unsigned int dtype) {
        return dtype == Type.ComplexDouble || dtype == Type.ComplexFloat;
      }

      // The tolerance of the checks for a matrix of this dtype with entries of order one.
      inline double tolerance(unsigned int dtype) {
        return (dtype == Type.Float || dtype == Type.ComplexFloat) ? 5e-5 : 1e-11;
      }

      // The tridiagonal matrix with `diagonal` on the diagonal and `hopping` above it (its
      // conjugate below). Eigenvalues: diagonal + 2 |hopping| cos(k pi / (n + 1)), k = 1 .. n,
      // all distinct.
      inline Matrix tridiagonal(std::size_t n, double diagonal, std::complex<double> hopping) {
        Matrix m(n);
        for (std::size_t i = 0; i < n; i++) {
          m(i, i) = diagonal;
          if (i + 1 < n) {
            m(i, i + 1) = hopping;
            m(i + 1, i) = std::conj(hopping);
          }
        }
        return m;
      }
      inline std::vector<double> tridiagonal_spectrum(std::size_t n, double diagonal,
                                                      std::complex<double> hopping) {
        std::vector<double> values(n);
        for (std::size_t k = 1; k <= n; k++)
          values[k - 1] =
            diagonal + 2. * std::abs(hopping) * std::cos(k * std::numbers::pi / (n + 1.));
        return values;
      }

      // The real symmetric ring of n >= 3 sites with hopping t. Eigenvalues:
      // 2 t cos(2 pi k / n), k = 0 .. n - 1, which are two-fold degenerate except for k = 0 and,
      // for even n, k = n / 2.
      inline Matrix ring(std::size_t n, double hopping) {
        Matrix m(n);
        for (std::size_t i = 0; i < n; i++) {
          m(i, (i + 1) % n) = hopping;
          m((i + 1) % n, i) = hopping;
        }
        return m;
      }
      inline std::vector<double> ring_spectrum(std::size_t n, double hopping) {
        std::vector<double> values(n);
        for (std::size_t k = 0; k < n; k++)
          values[k] = 2. * hopping * std::cos(2. * std::numbers::pi * k / n);
        return values;
      }

      // shift * 1 + weight * u u^dagger, with u_j = (j + 1) exp(i phase j). Eigenvalues: shift,
      // (n - 1)-fold degenerate, and shift + weight |u|^2 with |u|^2 = n (n + 1) (2 n + 1) / 6.
      inline Matrix rank_one_update(std::size_t n, double shift, double weight, double phase) {
        std::vector<std::complex<double>> u(n);
        for (std::size_t j = 0; j < n; j++) u[j] = std::polar(j + 1., phase * j);
        Matrix m(n);
        for (std::size_t i = 0; i < n; i++) {
          for (std::size_t j = 0; j < n; j++) m(i, j) = weight * u[i] * std::conj(u[j]);
          m(i, i) += shift;
        }
        return m;
      }
      inline std::vector<double> rank_one_update_spectrum(std::size_t n, double shift,
                                                          double weight) {
        std::vector<double> values(n, shift);
        values.back() = shift + weight * (n * (n + 1.) * (2. * n + 1.) / 6.);
        return values;
      }

      // H m H with the Householder reflection H = 1 - 2 w w^dagger / |w|^2,
      // w_j = exp(i phase j) / (j + 1): a dense matrix with the spectrum of m.
      inline Matrix reflected(const Matrix &m, double phase) {
        const std::size_t n = m.n;
        std::vector<std::complex<double>> w(n);
        double norm2 = 0;
        for (std::size_t j = 0; j < n; j++) {
          w[j] = std::polar(1. / (j + 1.), phase * j);
          norm2 += std::norm(w[j]);
        }
        Matrix h(n);
        for (std::size_t i = 0; i < n; i++) {
          for (std::size_t j = 0; j < n; j++)
            h(i, j) = (i == j ? 1. : 0.) - 2. * w[i] * std::conj(w[j]) / norm2;
        }
        Matrix hm(n), out(n);
        for (std::size_t i = 0; i < n; i++)
          for (std::size_t j = 0; j < n; j++)
            for (std::size_t k = 0; k < n; k++) hm(i, j) += h(i, k) * m(k, j);
        for (std::size_t i = 0; i < n; i++)
          for (std::size_t j = 0; j < n; j++)
            for (std::size_t k = 0; k < n; k++) out(i, j) += hm(i, k) * h(k, j);
        return out;
      }

      // The matrix as a Tensor of the given dtype on the CPU. For a real dtype the matrix must
      // be real.
      inline Tensor to_tensor(const Matrix &m, unsigned int dtype) {
        Tensor out({m.n, m.n}, dtype);
        for (std::size_t i = 0; i < m.n; i++) {
          for (std::size_t j = 0; j < m.n; j++) {
            const std::complex<double> value = m(i, j);
            if (dtype == Type.ComplexDouble) {
              out.at<cytnx_complex128>({i, j}) = value;
            } else if (dtype == Type.ComplexFloat) {
              out.at<cytnx_complex64>({i, j}) = cytnx_complex64(value);
            } else if (dtype == Type.Double) {
              out.at<cytnx_double>({i, j}) = value.real();
            } else {
              out.at<cytnx_float>({i, j}) = static_cast<float>(value.real());
            }
          }
        }
        return out;
      }

      // The elements of a rank-2 Tensor of a floating dtype, on any device.
      inline Matrix to_matrix(const Tensor &tensor) {
        const Tensor host = tensor.to(Device.cpu).contiguous();
        Matrix m(host.shape()[0]);
        const unsigned int dtype = host.dtype();
        for (std::size_t i = 0; i < m.data.size(); i++) {
          if (dtype == Type.ComplexDouble) {
            m.data[i] = host.storage().at<cytnx_complex128>(i);
          } else if (dtype == Type.ComplexFloat) {
            m.data[i] = std::complex<double>(host.storage().at<cytnx_complex64>(i));
          } else if (dtype == Type.Double) {
            m.data[i] = host.storage().at<cytnx_double>(i);
          } else {
            m.data[i] = host.storage().at<cytnx_float>(i);
          }
        }
        return m;
      }

      // The elements of a rank-1 Tensor of dtype Double or Float, on any device.
      inline std::vector<double> to_values(const Tensor &tensor) {
        const Tensor host = tensor.to(Device.cpu).contiguous();
        std::vector<double> values(host.shape()[0]);
        for (std::size_t i = 0; i < values.size(); i++) {
          values[i] = host.dtype() == Type.Double ? host.storage().at<cytnx_double>(i)
                                                  : host.storage().at<cytnx_float>(i);
        }
        return values;
      }

      // Checks eigenvalues against their expected values (given in any order): the right
      // dtype, the right count, ascending, equal to the expectation.
      inline void expect_eigenvalues(const Tensor &e, unsigned int matrix_dtype,
                                     std::vector<double> expected, double tol,
                                     const std::string &context) {
        SCOPED_TRACE(context);
        EXPECT_EQ(e.dtype(), Type.to_real(matrix_dtype));
        ASSERT_EQ(e.shape(), std::vector<cytnx_uint64>({expected.size()}));
        const std::vector<double> values = to_values(e);
        std::sort(expected.begin(), expected.end());
        for (std::size_t k = 0; k < values.size(); k++) {
          EXPECT_NEAR(values[k], expected[k], tol) << "eigenvalue " << k;
          if (k > 0) EXPECT_LE(values[k - 1], values[k]) << "eigenvalue " << k;
        }
      }

      // Checks the eigenvectors, stored as the columns of `v`, of the matrix `a` with the
      // eigenvalues `e`, without reference to a particular basis:
      // the residual A v_k - e_k v_k, the orthonormality V^dagger V = 1, and the reconstruction
      // V diag(e) V^dagger = A.
      inline void expect_eigenvectors(const Matrix &a, const Tensor &e, const Tensor &v,
                                      unsigned int matrix_dtype, double tol,
                                      const std::string &context) {
        SCOPED_TRACE(context);
        const std::size_t n = a.n;
        EXPECT_EQ(v.dtype(), matrix_dtype);
        ASSERT_EQ(v.shape(), std::vector<cytnx_uint64>({n, n}));
        ASSERT_EQ(e.shape(), std::vector<cytnx_uint64>({n}));
        const Matrix vm = to_matrix(v);
        const std::vector<double> values = to_values(e);

        double residual = 0, orthonormality = 0, reconstruction = 0;
        for (std::size_t i = 0; i < n; i++) {
          for (std::size_t k = 0; k < n; k++) {
            std::complex<double> av = 0, overlap = 0, rebuilt = 0;
            for (std::size_t j = 0; j < n; j++) {
              av += a(i, j) * vm(j, k);
              overlap += std::conj(vm(j, i)) * vm(j, k);
              rebuilt += vm(i, j) * values[j] * std::conj(vm(k, j));
            }
            residual = std::max(residual, std::abs(av - values[k] * vm(i, k)));
            orthonormality = std::max(orthonormality, std::abs(overlap - (i == k ? 1. : 0.)));
            reconstruction = std::max(reconstruction, std::abs(rebuilt - a(i, k)));
          }
        }
        EXPECT_LT(residual, tol) << "max |A v - e v|";
        EXPECT_LT(orthonormality, tol) << "max |V^dagger V - 1|";
        EXPECT_LT(reconstruction, tol) << "max |V diag(e) V^dagger - A|";
      }

      // The full check of linalg::Eigh(A, true, false, ...) on a Tensor.
      inline void expect_eigh(const Matrix &a, const std::vector<Tensor> &out, unsigned int dtype,
                              const std::vector<double> &expected, double tol,
                              const std::string &context) {
        ASSERT_EQ(out.size(), 2u) << context;
        expect_eigenvalues(out[0], dtype, expected, tol, context);
        expect_eigenvectors(a, out[0], out[1], dtype, tol, context);
      }

    }  // namespace eigh
  }  // namespace test
}  // namespace cytnx

#endif  // CYTNX_TESTS_LINALG_TEST_EIGH_TEST_TOOLS_H_
