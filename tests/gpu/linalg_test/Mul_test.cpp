#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

namespace cytnx {
  namespace gpu_test {
    namespace {

      ::testing::AssertionResult CheckMulResult(const Tensor& gpu_result, const Tensor& left_tensor,
                                                const Tensor& right_tensor);

      ::testing::AssertionResult CheckMulScalarResult(const Tensor& gpu_result,
                                                      const Tensor& tensor,
                                                      const cytnx_double& scalar,
                                                      bool scalar_left = false);

      std::vector<std::vector<cytnx_uint64>> GetTestShapes();

      static cytnx_double GetTolerance(const unsigned int& dtype);

      class MulTestAllShapes : public ::testing::TestWithParam<std::vector<cytnx_uint64>> {};

      // Test tensor-to-tensor multiplication
      TEST_P(MulTestAllShapes, GpuTensorMulTensorAllTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto dtype : dtype_list) {
          if (dtype == Type.Bool) {
            continue;
          }

          SCOPED_TRACE("Testing Mul(tensor, tensor) with shape: " +
                       ::testing::PrintToString(shape) + " and dtype: " + std::to_string(dtype));

          Tensor gpu_tensor1 = Tensor(shape, dtype).to(Device.cuda);
          Tensor gpu_tensor2 = Tensor(shape, dtype).to(Device.cuda);
          InitTensorUniform(gpu_tensor1);
          InitTensorUniform(gpu_tensor2);

          Tensor gpu_result = linalg::Mul(gpu_tensor1, gpu_tensor2);
          EXPECT_TRUE(CheckMulResult(gpu_result, gpu_tensor1, gpu_tensor2));

          Tensor gpu_result_member = gpu_tensor1.Mul(gpu_tensor2);
          EXPECT_TRUE(CheckMulResult(gpu_result_member, gpu_tensor1, gpu_tensor2));

          Tensor gpu_result_operator = gpu_tensor1 * gpu_tensor2;
          EXPECT_TRUE(CheckMulResult(gpu_result_operator, gpu_tensor1, gpu_tensor2));
        }
      }

      // Test scalar-to-tensor multiplication
      TEST_P(MulTestAllShapes, GpuScalarMulTensorAllTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto dtype : dtype_list) {
          if (dtype == Type.Bool) {
            continue;
          }

          SCOPED_TRACE("Testing Mul(scalar, tensor) with shape: " +
                       ::testing::PrintToString(shape) + " and dtype: " + std::to_string(dtype));

          Tensor gpu_tensor = Tensor(shape, dtype).to(Device.cuda);
          InitTensorUniform(gpu_tensor);
          cytnx_double scalar = 2.3;

          Tensor gpu_result = linalg::Mul(scalar, gpu_tensor);
          EXPECT_TRUE(CheckMulScalarResult(gpu_result, gpu_tensor, scalar, true));

          Tensor gpu_result_operator = scalar * gpu_tensor;
          EXPECT_TRUE(CheckMulScalarResult(gpu_result_operator, gpu_tensor, scalar, true));
        }
      }

      // Test tensor-to-scalar multiplication
      TEST_P(MulTestAllShapes, GpuTensorMulScalarAllTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto dtype : dtype_list) {
          if (dtype == Type.Bool) {
            continue;
          }

          SCOPED_TRACE("Testing Mul(tensor, scalar) with shape: " +
                       ::testing::PrintToString(shape) + " and dtype: " + std::to_string(dtype));

          Tensor gpu_tensor = Tensor(shape, dtype).to(Device.cuda);
          InitTensorUniform(gpu_tensor);
          cytnx_double scalar = 2.3;

          Tensor gpu_result = linalg::Mul(gpu_tensor, scalar);
          EXPECT_TRUE(CheckMulScalarResult(gpu_result, gpu_tensor, scalar, false));

          Tensor gpu_result_member = gpu_tensor.Mul(scalar);
          EXPECT_TRUE(CheckMulScalarResult(gpu_result_member, gpu_tensor, scalar, false));

          Tensor gpu_result_operator = gpu_tensor * scalar;
          EXPECT_TRUE(CheckMulScalarResult(gpu_result_operator, gpu_tensor, scalar, false));
        }
      }

      // Test in-place tensor multiplication
      TEST_P(MulTestAllShapes, GpuTensorImulAllTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto dtype : dtype_list) {
          if (dtype == Type.Bool) {
            continue;
          }

          SCOPED_TRACE("Testing iMul(tensor, tensor) with shape: " +
                       ::testing::PrintToString(shape) + " and dtype: " + std::to_string(dtype));

          Tensor gpu_tensor1 = Tensor(shape, dtype).to(Device.cuda);
          Tensor gpu_tensor2 = Tensor(shape, dtype).to(Device.cuda);
          InitTensorUniform(gpu_tensor1);
          InitTensorUniform(gpu_tensor2);

          Tensor original_gpu_tensor1 = gpu_tensor1.clone();
          Tensor original_gpu_tensor2 = gpu_tensor2.clone();

          linalg::iMul(gpu_tensor1, gpu_tensor2);
          EXPECT_TRUE(CheckMulResult(gpu_tensor1, original_gpu_tensor1, original_gpu_tensor2));

          Tensor gpu_tensor1_op = original_gpu_tensor1.clone();
          gpu_tensor1_op *= original_gpu_tensor2;
          EXPECT_TRUE(CheckMulResult(gpu_tensor1_op, original_gpu_tensor1, original_gpu_tensor2));
        }
      }

      TEST(MulMixedDtypeTest, GpuTensorMulTensorMixedUnsignedSignedTypePromote) {
        Tensor lhs = arange(0, 6, 1, Type.Uint32).reshape({2, 3});
        Tensor rhs = arange(0, 6, 1, Type.Int16).reshape({2, 3});
        lhs = lhs.to(Device.cuda);
        rhs = rhs.to(Device.cuda);

        Tensor gpu_result = linalg::Mul(lhs, rhs);
        Tensor expected_cpu = linalg::Mul(lhs.to(Device.cpu), rhs.to(Device.cpu));
        Tensor gpu_result_cpu = gpu_result.to(Device.cpu);

        EXPECT_EQ(gpu_result.dtype(), expected_cpu.dtype());
        EXPECT_TRUE(AreNearlyEqTensor(gpu_result_cpu, expected_cpu, 1e-6));
      }

      TEST(MulMixedDtypeTest, GpuScalarMulTensorMixedUnsignedSignedTypePromote) {
        const cytnx_uint32 scalar = 5;
        Tensor rhs = arange(0, 6, 1, Type.Int16, Device.cuda).reshape({2, 3});

        Tensor gpu_result_l = linalg::Mul(scalar, rhs);
        Tensor gpu_result_r = linalg::Mul(rhs, scalar);

        Tensor rhs_cpu = rhs.to(Device.cpu);
        Tensor expected_l = linalg::Mul(scalar, rhs_cpu);
        Tensor expected_r = linalg::Mul(rhs_cpu, scalar);
        Tensor gpu_result_l_cpu = gpu_result_l.to(Device.cpu);
        Tensor gpu_result_r_cpu = gpu_result_r.to(Device.cpu);

        EXPECT_EQ(gpu_result_l.dtype(), expected_l.dtype());
        EXPECT_EQ(gpu_result_r.dtype(), expected_r.dtype());
        EXPECT_TRUE(AreNearlyEqTensor(gpu_result_l_cpu, expected_l, 1e-6));
        EXPECT_TRUE(AreNearlyEqTensor(gpu_result_r_cpu, expected_r, 1e-6));
      }

      INSTANTIATE_TEST_SUITE_P(MulTests, MulTestAllShapes, ::testing::ValuesIn(GetTestShapes()));

      ::testing::AssertionResult CheckMulResult(const Tensor& gpu_result, const Tensor& left_tensor,
                                                const Tensor& right_tensor) {
        // Compare CUDA Mul result against CPU Mul result
        Tensor left_cpu = left_tensor.to(Device.cpu);
        Tensor right_cpu = right_tensor.to(Device.cpu);
        Tensor expected_cpu = linalg::Mul(left_cpu, right_cpu);
        Tensor gpu_result_cpu = gpu_result.to(Device.cpu);

        cytnx_double tolerance = GetTolerance(gpu_result.dtype());

        if (!AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
          return ::testing::AssertionFailure()
                 << "Mul result mismatch: CUDA Mul result differs from CPU Mul result. "
                 << "Left dtype: " << left_tensor.dtype()
                 << ", Right dtype: " << right_tensor.dtype() << ", tolerance used: " << tolerance;
        }

        return ::testing::AssertionSuccess();
      }

      ::testing::AssertionResult CheckMulScalarResult(const Tensor& gpu_result,
                                                      const Tensor& tensor,
                                                      const cytnx_double& scalar,
                                                      bool scalar_left) {
        // Compare CUDA Mul result against CPU Mul result
        Tensor tensor_cpu = tensor.to(Device.cpu);
        Tensor expected_cpu;

        if (scalar_left) {
          expected_cpu = linalg::Mul(scalar, tensor_cpu);
        } else {
          expected_cpu = linalg::Mul(tensor_cpu, scalar);
        }

        Tensor gpu_result_cpu = gpu_result.to(Device.cpu);

        cytnx_double tolerance = GetTolerance(gpu_result.dtype());

        if (!AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
          return ::testing::AssertionFailure()
                 << "Mul scalar result mismatch: CUDA Mul result differs from CPU Mul result. "
                 << "Tensor dtype: " << tensor.dtype() << ", scalar: " << scalar
                 << ", scalar_left: " << scalar_left << ", tolerance used: " << tolerance;
        }

        return ::testing::AssertionSuccess();
      }

      std::vector<std::vector<cytnx_uint64>> GetTestShapes() {
        std::vector<std::vector<cytnx_uint64>> all_shapes;

        auto shapes_1d = GenerateTestShapes(1, 1, 1024, 4);
        auto shapes_2d = GenerateTestShapes(2, 1, 512, 4);
        auto shapes_3d = GenerateTestShapes(3, 1, 64, 4);
        auto shapes_4d = GenerateTestShapes(4, 1, 32, 4);

        all_shapes.insert(all_shapes.end(), shapes_1d.begin(), shapes_1d.end());
        all_shapes.insert(all_shapes.end(), shapes_2d.begin(), shapes_2d.end());
        all_shapes.insert(all_shapes.end(), shapes_3d.begin(), shapes_3d.end());
        all_shapes.insert(all_shapes.end(), shapes_4d.begin(), shapes_4d.end());

        return all_shapes;
      }

      static cytnx_double GetTolerance(const unsigned int& dtype) {
        cytnx_double tolerance = 1e-6;
        if (dtype == Type.ComplexFloat) {
          tolerance = 0.1;
        }
        return tolerance;
      }

      // Non-contiguous tensor(x)tensor on the GPU (#1003, #988): the GPU front end used to
      // reject a non-contiguous operand ("must be contiguous"); it now feeds the layout to
      // the shared dispatch kernel like Add/Sub. Both operands are permuted so both operand
      // offsets are gathered through the inverse mappers. Compare against the independent CPU
      // path across all dtypes. Inputs are built with CPU arange then moved to the GPU.
      TEST(MulNonContig, GpuNoncontiguousTensorTensorMatchesCpu) {
        for (auto dtype : dtype_list) {
          if (dtype == Type.Bool) continue;
          SCOPED_TRACE("Mul non-contiguous dtype=" + std::to_string(dtype));

          Tensor gpu_l = arange(1, 7, 1, dtype).reshape({3, 2}).permute({1, 0}).to(Device.cuda);
          Tensor gpu_r = arange(2, 8, 1, dtype).reshape({3, 2}).permute({1, 0}).to(Device.cuda);
          ASSERT_FALSE(gpu_l.is_contiguous());
          ASSERT_FALSE(gpu_r.is_contiguous());

          Tensor gpu_out = linalg::Mul(gpu_l, gpu_r);
          Tensor cpu_out = linalg::Mul(gpu_l.to(Device.cpu), gpu_r.to(Device.cpu));
          EXPECT_EQ(gpu_out.dtype(), cpu_out.dtype());
          EXPECT_TRUE(
            AreNearlyEqTensor(gpu_out.to(Device.cpu), cpu_out, GetTolerance(gpu_out.dtype())));
        }
      }

      // Independent hand-computed literal (NOT the CPU oracle) for the non-contiguous gather:
      // l logical = [[1,3,5],[2,4,6]], r logical = [[10,30,50],[20,40,60]] (arange 3x2
      // permuted), so l*r = [[10,90,250],[40,160,360]] -- distinct per element, so a wrong
      // multi-index decomposition (not just a wrong L/R pairing) is caught.
      TEST(MulNonContig, GpuNoncontiguousLiteral) {
        Tensor l = arange(1, 7, 1, Type.Int64).reshape({3, 2}).permute({1, 0}).to(Device.cuda);
        Tensor r = arange(10, 70, 10, Type.Int64).reshape({3, 2}).permute({1, 0}).to(Device.cuda);
        ASSERT_FALSE(l.is_contiguous());
        Tensor got = linalg::Mul(l, r).to(Device.cpu);
        EXPECT_EQ(got.dtype(), Type.Int64);
        const cytnx_int64 expect[2][3] = {{10, 90, 250}, {40, 160, 360}};
        for (cytnx_uint64 i = 0; i < 2; i++)
          for (cytnx_uint64 j = 0; j < 3; j++) EXPECT_EQ(got.at<cytnx_int64>({i, j}), expect[i][j]);
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
