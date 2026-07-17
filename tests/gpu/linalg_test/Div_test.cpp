#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

namespace cytnx {
  namespace gpu_test {
    namespace {

      static cytnx_double GetTolerance(const unsigned int& dtype) {
        cytnx_double tolerance;
        if (dtype == Type.Float || dtype == Type.ComplexFloat) {
          tolerance = 1e-5;
        } else {
          tolerance = 1e-10;
        }
        return tolerance;
      }

      ::testing::AssertionResult CheckDivResult(const Tensor& gpu_result, const Tensor& left_tensor,
                                                const Tensor& right_tensor) {
        // Compare CUDA Div result against CPU Div result
        Tensor left_cpu = left_tensor.to(Device.cpu);
        Tensor right_cpu = right_tensor.to(Device.cpu);
        Tensor expected_cpu = linalg::Div(left_cpu, right_cpu);
        Tensor gpu_result_cpu = gpu_result.to(Device.cpu);

        cytnx_double tolerance = GetTolerance(gpu_result.dtype());

        if (!AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
          return ::testing::AssertionFailure()
                 << "Div result mismatch: CUDA Div result differs from CPU Div result. "
                 << "Left dtype: " << left_tensor.dtype()
                 << ", Right dtype: " << right_tensor.dtype() << ", tolerance used: " << tolerance;
        }

        return ::testing::AssertionSuccess();
      }

      ::testing::AssertionResult CheckDivScalarResult(const Tensor& gpu_result,
                                                      const Tensor& tensor,
                                                      const cytnx_double& scalar,
                                                      bool scalar_left) {
        // Compare CUDA Div result against CPU Div result
        Tensor tensor_cpu = tensor.to(Device.cpu);
        Tensor expected_cpu;

        if (scalar_left) {
          expected_cpu = linalg::Div(scalar, tensor_cpu);
        } else {
          expected_cpu = linalg::Div(tensor_cpu, scalar);
        }

        Tensor gpu_result_cpu = gpu_result.to(Device.cpu);

        cytnx_double tolerance = GetTolerance(gpu_result.dtype());

        if (!AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
          return ::testing::AssertionFailure()
                 << "Div scalar result mismatch: CUDA Div result differs from CPU Div result. "
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

      class DivTestAllShapes : public ::testing::TestWithParam<std::vector<cytnx_uint64>> {};

      // Test tensor-to-tensor division
      TEST_P(DivTestAllShapes, GpuTensorDivTensorAllTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto dtype : dtype_list) {
          if (dtype == Type.Bool) {
            continue;
          }

          SCOPED_TRACE("Testing Div(tensor, tensor) with shape: " +
                       ::testing::PrintToString(shape) + " and dtype: " + std::to_string(dtype));

          Tensor gpu_tensor1 = Tensor(shape, dtype, Device.cuda);
          Tensor gpu_tensor2 = Tensor(shape, dtype, Device.cuda);
          InitTensorUniform(gpu_tensor1);
          InitTensorUniform(gpu_tensor2);
          // Add small offset to avoid division by zero
          gpu_tensor1 = gpu_tensor1 + 1.0;
          gpu_tensor2 = gpu_tensor2 + 1.0;

          Tensor gpu_result = linalg::Div(gpu_tensor1, gpu_tensor2);
          EXPECT_TRUE(CheckDivResult(gpu_result, gpu_tensor1, gpu_tensor2));

          Tensor gpu_result_member = gpu_tensor1.Div(gpu_tensor2);
          EXPECT_TRUE(CheckDivResult(gpu_result_member, gpu_tensor1, gpu_tensor2));

          Tensor gpu_result_operator = gpu_tensor1 / gpu_tensor2;
          EXPECT_TRUE(CheckDivResult(gpu_result_operator, gpu_tensor1, gpu_tensor2));
        }
      }

      // Test scalar-to-tensor division
      TEST_P(DivTestAllShapes, GpuScalarDivTensorAllTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto dtype : dtype_list) {
          if (dtype == Type.Bool) {
            continue;
          }

          SCOPED_TRACE("Testing Div(scalar, tensor) with shape: " +
                       ::testing::PrintToString(shape) + " and dtype: " + std::to_string(dtype));

          Tensor gpu_tensor = Tensor(shape, dtype, Device.cuda);
          InitTensorUniform(gpu_tensor);
          // Add small offset to avoid division by zero
          gpu_tensor = gpu_tensor + 1.0;
          cytnx_double scalar = 10.0;

          Tensor gpu_result = linalg::Div(scalar, gpu_tensor);
          EXPECT_TRUE(CheckDivScalarResult(gpu_result, gpu_tensor, scalar, true));

          Tensor gpu_result_operator = scalar / gpu_tensor;
          EXPECT_TRUE(CheckDivScalarResult(gpu_result_operator, gpu_tensor, scalar, true));
        }
      }

      // Test tensor-to-scalar division
      TEST_P(DivTestAllShapes, GpuTensorDivScalarAllTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto dtype : dtype_list) {
          if (dtype == Type.Bool) {
            continue;
          }

          SCOPED_TRACE("Testing Div(tensor, scalar) with shape: " +
                       ::testing::PrintToString(shape) + " and dtype: " + std::to_string(dtype));

          Tensor gpu_tensor = Tensor(shape, dtype, Device.cuda);
          InitTensorUniform(gpu_tensor);
          gpu_tensor = gpu_tensor + 1.0;
          cytnx_double scalar = 2.0;  // Use non-zero scalar

          Tensor gpu_result = linalg::Div(gpu_tensor, scalar);
          EXPECT_TRUE(CheckDivScalarResult(gpu_result, gpu_tensor, scalar, false));

          Tensor gpu_result_member = gpu_tensor.Div(scalar);
          EXPECT_TRUE(CheckDivScalarResult(gpu_result_member, gpu_tensor, scalar, false));

          Tensor gpu_result_operator = gpu_tensor / scalar;
          EXPECT_TRUE(CheckDivScalarResult(gpu_result_operator, gpu_tensor, scalar, false));
        }
      }

      // Test in-place tensor division
      TEST_P(DivTestAllShapes, GpuTensorIdivAllTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto dtype : dtype_list) {
          if (dtype == Type.Bool) {
            continue;
          }

          SCOPED_TRACE("Testing iDiv(tensor, tensor) with shape: " +
                       ::testing::PrintToString(shape) + " and dtype: " + std::to_string(dtype));

          Tensor gpu_tensor1 = Tensor(shape, dtype, Device.cuda);
          Tensor gpu_tensor2 = Tensor(shape, dtype, Device.cuda);
          InitTensorUniform(gpu_tensor1);
          InitTensorUniform(gpu_tensor2);
          // Add small offset to avoid division by zero
          gpu_tensor1 = gpu_tensor1 + 1.0;
          gpu_tensor2 = gpu_tensor2 + 1.0;

          Tensor original_gpu_tensor1 = gpu_tensor1.clone();
          Tensor original_gpu_tensor2 = gpu_tensor2.clone();

          linalg::iDiv(gpu_tensor1, gpu_tensor2);
          EXPECT_TRUE(CheckDivResult(gpu_tensor1, original_gpu_tensor1, original_gpu_tensor2));

          Tensor gpu_tensor1_op = original_gpu_tensor1.clone();
          gpu_tensor1_op /= original_gpu_tensor2;
          EXPECT_TRUE(CheckDivResult(gpu_tensor1_op, original_gpu_tensor1, original_gpu_tensor2));
        }
      }

      TEST(DivMixedDtypeTest, GpuTensorDivTensorMixedUnsignedSignedTypePromote) {
        Tensor lhs = arange(1, 7, 1, Type.Uint32).reshape({2, 3});
        Tensor rhs = arange(1, 7, 1, Type.Int16).reshape({2, 3});
        lhs = lhs.to(Device.cuda);
        rhs = rhs.to(Device.cuda);

        Tensor gpu_result = linalg::Div(lhs, rhs);
        Tensor expected_cpu = linalg::Div(lhs.to(Device.cpu), rhs.to(Device.cpu));
        Tensor gpu_result_cpu = gpu_result.to(Device.cpu);

        EXPECT_EQ(gpu_result.dtype(), expected_cpu.dtype());
        EXPECT_TRUE(
          AreNearlyEqTensor(gpu_result_cpu, expected_cpu, GetTolerance(gpu_result.dtype())));
      }

      TEST(DivMixedDtypeTest, GpuScalarDivTensorMixedUnsignedSignedTypePromote) {
        const cytnx_uint32 scalar = 12;
        Tensor rhs = arange(1, 7, 1, Type.Int16, Device.cuda).reshape({2, 3});

        Tensor gpu_result_l = linalg::Div(scalar, rhs);
        Tensor gpu_result_r = linalg::Div(rhs, scalar);

        Tensor rhs_cpu = rhs.to(Device.cpu);
        Tensor expected_l = linalg::Div(scalar, rhs_cpu);
        Tensor expected_r = linalg::Div(rhs_cpu, scalar);
        Tensor gpu_result_l_cpu = gpu_result_l.to(Device.cpu);
        Tensor gpu_result_r_cpu = gpu_result_r.to(Device.cpu);

        EXPECT_EQ(gpu_result_l.dtype(), expected_l.dtype());
        EXPECT_EQ(gpu_result_r.dtype(), expected_r.dtype());
        EXPECT_TRUE(
          AreNearlyEqTensor(gpu_result_l_cpu, expected_l, GetTolerance(gpu_result_l.dtype())));
        EXPECT_TRUE(
          AreNearlyEqTensor(gpu_result_r_cpu, expected_r, GetTolerance(gpu_result_r.dtype())));
      }

      INSTANTIATE_TEST_SUITE_P(DivTests, DivTestAllShapes, ::testing::ValuesIn(GetTestShapes()));

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
