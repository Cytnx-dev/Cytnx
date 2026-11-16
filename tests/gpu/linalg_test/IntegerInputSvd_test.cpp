#include <algorithm>
#include <cmath>
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

      // Integer and Bool input to the truncated SVD (Gesvd_truncate, Rsvd), issue #1274: the
      // input is converted to Double, so S, U and vT come back as Double.
      //
      // The input is a direct sum of three constant blocks, block k being coefficient_k times a
      // rows_k x cols_k block of ones. Each block has rank one with the singular value
      // |coefficient_k| sqrt(rows_k cols_k), so the matrix has these three singular values and
      // zeros, all its entries are integers, and truncating to `kept` values leaves the first
      // `kept` blocks. None of the expected values is taken from another Cytnx code path.
      struct ConstantBlock {
        std::size_t rows;
        std::size_t cols;
        int coefficient;
      };

      // Negative and non-unit entries where the dtype can hold them.
      template <typename T>
      std::vector<int> block_coefficients() {
        if constexpr (std::is_same_v<T, cytnx_bool>) {
          return {1, 1, 1};
        } else if constexpr (std::is_signed_v<T>) {
          return {5, -3, 2};
        } else {
          return {5, 3, 2};
        }
      }

      // Row-major direct sum of the first `kept` blocks, zero elsewhere.
      std::vector<double> block_matrix(const std::vector<ConstantBlock>& blocks, std::size_t rows,
                                       std::size_t cols, std::size_t kept) {
        std::vector<double> out(rows * cols, 0.);
        std::size_t row_offset = 0, col_offset = 0;
        for (std::size_t k = 0; k < blocks.size(); k++) {
          if (k < kept) {
            for (std::size_t i = 0; i < blocks[k].rows; i++) {
              for (std::size_t j = 0; j < blocks[k].cols; j++) {
                out[(row_offset + i) * cols + col_offset + j] = blocks[k].coefficient;
              }
            }
          }
          row_offset += blocks[k].rows;
          col_offset += blocks[k].cols;
        }
        return out;
      }

      template <typename T>
      Tensor make_matrix(const std::vector<double>& values, std::size_t rows, std::size_t cols,
                         int device) {
        Tensor out({rows, cols}, Type.cy_typeid_v<T>);
        T* data = out.ptr_as<T>();
        for (std::size_t i = 0; i < values.size(); i++) data[i] = static_cast<T>(values[i]);
        return out.to(device);
      }

      // Copies the elements to the host. The dtype must already have been checked to be Double.
      std::vector<double> host_values(const Tensor& tensor) {
        Tensor host = tensor.to(Device.cpu).contiguous();
        const double* data = host.ptr_as<double>();
        return std::vector<double>(data, data + host.storage().size());
      }

      const double kTolerance = 1e-10;

      // Checks `out` = [S, U, vT] or [S, U, vT, terr] against a matrix with the given singular
      // values of which `kept` survive.
      void check_truncated_svd(std::vector<Tensor> out, std::size_t rows, std::size_t cols,
                               const std::vector<double>& singular_values, std::size_t kept,
                               unsigned int return_err, int device,
                               const std::vector<double>& expected_truncated_matrix) {
        ASSERT_EQ(out.size(), return_err ? 4u : 3u);
        for (const Tensor& tensor : out) {
          EXPECT_EQ(tensor.device(), device);
          ASSERT_EQ(tensor.dtype(), Type.Double);
        }
        const Tensor& s = out[0];
        const Tensor& u = out[1];
        const Tensor& vt = out[2];
        ASSERT_EQ(s.shape(), std::vector<cytnx_uint64>({kept}));
        ASSERT_EQ(u.shape(), std::vector<cytnx_uint64>({rows, kept}));
        ASSERT_EQ(vt.shape(), std::vector<cytnx_uint64>({kept, cols}));

        const std::vector<double> s_host = host_values(s);
        for (std::size_t k = 0; k < kept; k++) {
          EXPECT_NEAR(s_host[k], singular_values[k], kTolerance) << "S[" << k << "]";
        }

        const std::vector<double> u_host = host_values(u);
        const std::vector<double> vt_host = host_values(vt);
        double max_residual = 0.;
        for (std::size_t i = 0; i < rows; i++) {
          for (std::size_t j = 0; j < cols; j++) {
            double element = 0.;
            for (std::size_t k = 0; k < kept; k++) {
              element += u_host[i * kept + k] * s_host[k] * vt_host[k * cols + j];
            }
            max_residual =
              std::max(max_residual, std::abs(element - expected_truncated_matrix[i * cols + j]));
          }
        }
        EXPECT_LT(max_residual, 10. * kTolerance)
          << "U diag(S) vT differs from the truncated matrix";

        if (!return_err) return;
        const std::vector<double> terr_host = host_values(out[3]);
        if (return_err == 1) {
          ASSERT_EQ(terr_host.size(), 1u);
          EXPECT_NEAR(terr_host[0], singular_values[kept], kTolerance);
        } else {
          const std::size_t discarded = singular_values.size() - kept;
          ASSERT_EQ(terr_host.size(), discarded);
          for (std::size_t k = 0; k < discarded; k++) {
            EXPECT_NEAR(terr_host[k], singular_values[kept + k], kTolerance) << "terr[" << k << "]";
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

      struct Cut {
        std::size_t keepdim;
        double err;
        std::size_t kept;  // expected number of kept singular values
      };

      // Every cut discards at least one singular value.
      const std::vector<Cut> kCuts = {
        {2, 0., 2},  // cut by keepdim
        {3, 0., 3},  // keepdim equals the rank
        {6, 0.5, 3},  // err cuts the zero singular values
        {6, 1e9, 1},  // err would cut everything, one value is always kept
      };

      // Oversampling that makes Rsvd take as many samples as there are singular values, so it runs
      // a full SVD without a random projection and its results are exact.
      const cytnx_uint64 kFullOversampling = 10;

      std::vector<Tensor> run(Function function, Container container, const Tensor& matrix,
                              const Cut& cut, unsigned int return_err) {
        if (container == Container::kTensor) {
          if (function == Function::kGesvdTruncate) {
            return linalg::Gesvd_truncate(matrix, cut.keepdim, cut.err, true, true, return_err);
          }
          return linalg::Rsvd(matrix, cut.keepdim, cut.err, true, true, return_err, 1,
                              kFullOversampling);
        }
        const UniTensor unitensor(matrix, false, 1);
        if (function == Function::kGesvdTruncate) {
          return blocks_of(
            linalg::Gesvd_truncate(unitensor, cut.keepdim, cut.err, true, true, return_err));
        }
        return blocks_of(linalg::Rsvd(unitensor, cut.keepdim, cut.err, true, true, return_err, 1,
                                      kFullOversampling));
      }

      // The host results are checked against the same expectations, which shows that the expected
      // values are the ones the CPU implementation returns.
      template <typename T>
      void check_integer_input(Function function, Container container) {
        SCOPED_TRACE(Type.getname(Type.cy_typeid_v<T>));
        const std::vector<int> c = block_coefficients<T>();
        const std::vector<std::vector<ConstantBlock>> layouts = {
          {{3, 4, c[0]}, {2, 3, c[1]}, {1, 1, c[2]}},  // 6 x 8
          {{4, 3, c[0]}, {3, 2, c[1]}, {1, 1, c[2]}},  // 8 x 6
          {{3, 3, c[0]}, {2, 2, c[1]}, {1, 1, c[2]}},  // 6 x 6
        };
        for (int device : {Device.cpu, Device.cuda}) {
          for (const std::vector<ConstantBlock>& blocks : layouts) {
            std::size_t rows = 0, cols = 0;
            for (const ConstantBlock& block : blocks) {
              rows += block.rows;
              cols += block.cols;
            }
            std::vector<double> singular_values(std::min(rows, cols), 0.);
            for (std::size_t k = 0; k < blocks.size(); k++) {
              singular_values[k] = std::abs(blocks[k].coefficient) *
                                   std::sqrt(static_cast<double>(blocks[k].rows * blocks[k].cols));
            }
            const Tensor matrix =
              make_matrix<T>(block_matrix(blocks, rows, cols, blocks.size()), rows, cols, device);
            for (const Cut& cut : kCuts) {
              const std::vector<double> expected = block_matrix(blocks, rows, cols, cut.kept);
              for (unsigned int return_err : {0u, 1u, 2u}) {
                // A cut by err without a returned error is left out: U from the cuQuantum path
                // is wrong in that mode for every dtype, which is issue #1250.
                if (cut.err > 0. && return_err == 0) continue;
                SCOPED_TRACE(std::string(device == Device.cpu ? "cpu" : "cuda") + ", shape " +
                             std::to_string(rows) + "x" + std::to_string(cols) + ", keepdim " +
                             std::to_string(cut.keepdim) + ", err " + std::to_string(cut.err) +
                             ", return_err " + std::to_string(return_err));
                check_truncated_svd(run(function, container, matrix, cut, return_err), rows, cols,
                                    singular_values, cut.kept, return_err, device, expected);
              }
            }
          }
        }
      }

      void check_all_integer_inputs(Function function, Container container) {
        check_integer_input<cytnx_int64>(function, container);
        check_integer_input<cytnx_uint64>(function, container);
        check_integer_input<cytnx_int32>(function, container);
        check_integer_input<cytnx_uint32>(function, container);
        check_integer_input<cytnx_int16>(function, container);
        check_integer_input<cytnx_uint16>(function, container);
        check_integer_input<cytnx_bool>(function, container);
      }

      TEST(IntegerInputSvd, GpuGesvdTruncateTensor) {
        check_all_integer_inputs(Function::kGesvdTruncate, Container::kTensor);
      }
      TEST(IntegerInputSvd, GpuGesvdTruncateDenseUniTensor) {
        check_all_integer_inputs(Function::kGesvdTruncate, Container::kDenseUniTensor);
      }
      TEST(IntegerInputSvd, GpuRsvdTensor) {
        check_all_integer_inputs(Function::kRsvd, Container::kTensor);
      }
      TEST(IntegerInputSvd, GpuRsvdDenseUniTensor) {
        check_all_integer_inputs(Function::kRsvd, Container::kDenseUniTensor);
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
