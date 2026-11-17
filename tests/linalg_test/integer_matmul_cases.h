#ifndef CYTNX_TESTS_LINALG_TEST_INTEGER_MATMUL_CASES_H_
#define CYTNX_TESTS_LINALG_TEST_INTEGER_MATMUL_CASES_H_

// Matrix products of integer and Bool tensors (#1246), shared by the CPU suite and the GPU suite.
// Every expected value is either written out by hand or computed by a plain loop over
// std::vector, never by another Cytnx code path.

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
  namespace integer_matmul_test {

    template <typename T>
    struct Matrix {
      std::size_t rows;
      std::size_t cols;
      std::vector<T> values;  // row-major
    };

    template <typename T>
    Tensor make_tensor(const std::vector<cytnx_uint64>& shape, const std::vector<T>& values,
                       int device) {
      Tensor out(shape, Type.cy_typeid_v<T>);
      std::copy(values.begin(), values.end(), out.ptr_as<T>());
      return out.to(device);
    }

    template <typename T>
    Tensor make_tensor(const Matrix<T>& matrix, int device) {
      return make_tensor<T>({matrix.rows, matrix.cols}, matrix.values, device);
    }

    template <typename T>
    void expect_values(const Tensor& result, const std::vector<cytnx_uint64>& shape,
                       const std::vector<T>& expected, int device) {
      EXPECT_EQ(result.dtype(), Type.cy_typeid_v<T>);
      EXPECT_EQ(result.device(), device);
      ASSERT_EQ(result.shape(), shape);
      const Tensor on_cpu = result.contiguous().to(Device.cpu);
      const T* data = on_cpu.ptr_as<T>();
      EXPECT_EQ(std::vector<T>(data, data + expected.size()), expected);
    }

    template <typename T>
    void expect_matrix(const Tensor& result, const Matrix<T>& expected, int device) {
      expect_values<T>(result, {expected.rows, expected.cols}, expected.values, device);
    }

    // One product L * R = product. In `wide` the contracted dimension (3) is smaller than the
    // number of output columns (4); in `narrow` it is larger (4 against 2).
    template <typename T>
    struct Product {
      Matrix<T> lhs;
      Matrix<T> rhs;
      Matrix<T> product;
    };

    template <typename T>
    Product<T> wide_unsigned() {
      return {{2, 3, {1, 2, 3, 4, 5, 6}},
              {3, 4, {1, 2, 3, 4, 0, 1, 2, 5, 7, 3, 1, 2}},
              {2, 4, {22, 13, 10, 20, 46, 31, 28, 53}}};
    }

    template <typename T>
    Product<T> narrow_unsigned() {
      return {{3, 4, {1, 2, 3, 0, 1, 4, 2, 3, 5, 0, 6, 2}},
              {4, 2, {2, 1, 3, 4, 5, 1, 6, 2}},
              {3, 2, {23, 12, 42, 25, 52, 15}}};
    }

    template <typename T>
    Product<T> wide_signed() {
      return {{2, 3, {1, -2, 3, -4, 5, 6}},
              {3, 4, {1, 2, -3, 4, 0, -1, 2, 5, 7, 3, 1, -2}},
              {2, 4, {22, 13, -4, -12, 38, 5, 28, -3}}};
    }

    template <typename T>
    Product<T> narrow_signed() {
      return {{3, 4, {1, -2, 3, 0, -1, 4, 2, -3, 5, 0, -6, 2}},
              {4, 2, {2, -1, 3, 4, -5, 1, 6, -2}},
              {3, 2, {-19, -6, -18, 25, 52, -15}}};
    }

    // Bool products are "or" over "and".
    inline Product<bool> wide_bool() {
      return {{2, 3, {true, false, false, false, false, true}},
              {3, 4, {false, true, false, true, true, true, true, true, true, false, false, false}},
              {2, 4, {false, true, false, true, true, false, false, false}}};
    }

    inline Product<bool> narrow_bool() {
      return {
        {3, 4, {false, true, false, false, false, false, false, false, true, false, true, false}},
        {4, 2, {true, false, false, true, false, false, true, true}},
        {3, 2, {false, true, false, false, true, false}}};
    }

    template <typename T>
    void check_matmul(const Product<T>& product, int device) {
      const Tensor result =
        linalg::Matmul(make_tensor(product.lhs, device), make_tensor(product.rhs, device));
      expect_matrix(result, product.product, device);
    }

    // The same product with the left operand stored transposed, so that Tensordot has to permute
    // it before the matrix product.
    template <typename T>
    void check_tensordot(const Product<T>& product, int device) {
      const Matrix<T>& lhs = product.lhs;
      Matrix<T> transposed{lhs.cols, lhs.rows, std::vector<T>(lhs.values.size())};
      for (std::size_t row = 0; row < lhs.rows; row++) {
        for (std::size_t col = 0; col < lhs.cols; col++) {
          transposed.values[col * lhs.rows + row] = lhs.values[row * lhs.cols + col];
        }
      }
      const Tensor result = linalg::Tensordot(make_tensor(transposed, device),
                                              make_tensor(product.rhs, device), {0}, {0});
      expect_matrix(result, product.product, device);
    }

    template <typename T>
    void check_contract(const Product<T>& product, int device) {
      UniTensor lhs(make_tensor(product.lhs, device), false, 1);
      UniTensor rhs(make_tensor(product.rhs, device), false, 1);
      lhs.set_labels(std::vector<std::string>{"row", "inner"});
      rhs.set_labels(std::vector<std::string>{"inner", "col"});
      const UniTensor result = Contract(lhs, rhs);
      ASSERT_EQ(result.labels(), (std::vector<std::string>{"row", "col"}));
      expect_matrix(result.get_block(), product.product, device);
    }

    // A product larger than one 512-thread block of the GPU kernel. The expected values come from
    // a plain triple loop; every entry and every partial sum stays within the range of Int16.
    template <typename T>
    void check_matmul_many_blocks(std::size_t rows, std::size_t inner, std::size_t cols,
                                  bool with_negative, int device) {
      const std::int64_t shift = with_negative ? 1 : 0;
      Matrix<T> lhs{rows, inner, std::vector<T>(rows * inner)};
      Matrix<T> rhs{inner, cols, std::vector<T>(inner * cols)};
      std::vector<std::int64_t> lhs_wide(rows * inner);
      std::vector<std::int64_t> rhs_wide(inner * cols);
      for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t c = 0; c < inner; c++) {
          lhs_wide[i * inner + c] = static_cast<std::int64_t>((i + 2 * c) % 3) - shift;
          lhs.values[i * inner + c] = static_cast<T>(lhs_wide[i * inner + c]);
        }
      }
      for (std::size_t c = 0; c < inner; c++) {
        for (std::size_t j = 0; j < cols; j++) {
          rhs_wide[c * cols + j] = static_cast<std::int64_t>((3 * c + j) % 5) - 2 * shift;
          rhs.values[c * cols + j] = static_cast<T>(rhs_wide[c * cols + j]);
        }
      }
      Matrix<T> expected{rows, cols, std::vector<T>(rows * cols)};
      for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
          std::int64_t sum = 0;
          for (std::size_t c = 0; c < inner; c++) {
            sum += lhs_wide[i * inner + c] * rhs_wide[c * cols + j];
          }
          expected.values[i * cols + j] = static_cast<T>(sum);
        }
      }
      const Tensor result = linalg::Matmul(make_tensor(lhs, device), make_tensor(rhs, device));
      expect_matrix(result, expected, device);
    }

    // Matmul_dg with the diagonal operand on the left, then on the right.
    template <typename T>
    void check_matmul_dg(const std::vector<T>& left_diagonal, const Matrix<T>& matrix,
                         const std::vector<T>& right_diagonal, const Matrix<T>& expected_left,
                         const Matrix<T>& expected_right, int device) {
      const Tensor dense = make_tensor(matrix, device);
      const Tensor from_left =
        linalg::Matmul_dg(make_tensor<T>({left_diagonal.size()}, left_diagonal, device), dense);
      expect_matrix(from_left, expected_left, device);
      const Tensor from_right =
        linalg::Matmul_dg(dense, make_tensor<T>({right_diagonal.size()}, right_diagonal, device));
      expect_matrix(from_right, expected_right, device);
    }

    template <typename T>
    void check_matmul_dg_unsigned(int device) {
      check_matmul_dg<T>({2, 3}, {2, 3, {1, 2, 3, 4, 5, 6}}, {2, 3, 4},
                         {2, 3, {2, 4, 6, 12, 15, 18}}, {2, 3, {2, 6, 12, 8, 15, 24}}, device);
    }

    template <typename T>
    void check_matmul_dg_signed(int device) {
      check_matmul_dg<T>({2, -3}, {2, 3, {1, -2, 3, -4, 5, 6}}, {2, -3, 4},
                         {2, 3, {2, -4, 6, 12, -15, -18}}, {2, 3, {2, 6, 12, -8, -15, 24}}, device);
    }

    inline void check_matmul_dg_bool(int device) {
      check_matmul_dg<bool>({true, false}, {2, 3, {true, false, true, true, true, false}},
                            {false, true, true}, {2, 3, {true, false, true, false, false, false}},
                            {2, 3, {false, false, true, false, true, false}}, device);
    }

    // Dot of a matrix with a vector, which uses the Matvec kernels.
    template <typename T>
    void check_dot(const Matrix<T>& matrix, const std::vector<T>& vector,
                   const std::vector<T>& expected, int device) {
      const Tensor result =
        linalg::Dot(make_tensor(matrix, device), make_tensor<T>({vector.size()}, vector, device));
      expect_values<T>(result, {matrix.rows}, expected, device);
    }

    // Runs every check that applies to the integer dtype T on `device`.
    template <typename T>
    void check_integer_dtype(int device) {
      constexpr bool is_signed = std::is_signed_v<T>;
      std::vector<Product<T>> products = {wide_unsigned<T>(), narrow_unsigned<T>()};
      if constexpr (is_signed) {
        products.push_back(wide_signed<T>());
        products.push_back(narrow_signed<T>());
      }
      for (std::size_t i = 0; i < products.size(); i++) {
        SCOPED_TRACE("product " + std::to_string(i));
        {
          SCOPED_TRACE("Matmul");
          check_matmul(products[i], device);
        }
        {
          SCOPED_TRACE("Tensordot");
          check_tensordot(products[i], device);
        }
        {
          SCOPED_TRACE("Contract");
          check_contract(products[i], device);
        }
      }
      for (const bool with_negative : {false, true}) {
        if (with_negative && !is_signed) continue;
        SCOPED_TRACE(with_negative ? "many blocks, signed" : "many blocks, unsigned");
        check_matmul_many_blocks<T>(40, 7, 30, with_negative, device);
        check_matmul_many_blocks<T>(30, 40, 20, with_negative, device);
      }
      {
        SCOPED_TRACE("Matmul_dg");
        check_matmul_dg_unsigned<T>(device);
        if constexpr (is_signed) check_matmul_dg_signed<T>(device);
      }
      {
        SCOPED_TRACE("Dot");
        check_dot<T>({2, 3, {1, 2, 3, 4, 5, 6}}, {2, 0, 5}, {17, 38}, device);
        if constexpr (is_signed) {
          check_dot<T>({2, 3, {1, -2, 3, -4, 5, 6}}, {2, -3, -1}, {5, -29}, device);
        }
      }
    }

    inline void check_bool_dtype(int device) {
      const std::vector<Product<bool>> products = {wide_bool(), narrow_bool()};
      for (std::size_t i = 0; i < products.size(); i++) {
        SCOPED_TRACE("product " + std::to_string(i));
        {
          SCOPED_TRACE("Matmul");
          check_matmul(products[i], device);
        }
        {
          SCOPED_TRACE("Tensordot");
          check_tensordot(products[i], device);
        }
      }
      {
        SCOPED_TRACE("Matmul_dg");
        check_matmul_dg_bool(device);
      }
      SCOPED_TRACE("Dot");
      check_dot<bool>({2, 3, {true, false, false, false, false, true}}, {false, true, true},
                      {false, true}, device);
    }

  }  // namespace integer_matmul_test
}  // namespace cytnx

#endif  // CYTNX_TESTS_LINALG_TEST_INTEGER_MATMUL_CASES_H_
