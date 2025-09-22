#include "gtest/gtest.h"

#include "gpu_test_tools.h"
#include "cytnx.hpp"

namespace DivTest {

  cytnx::cytnx_double GetTolerance(const unsigned int& dtype) {
    cytnx::cytnx_double tolerance;
    if (dtype == cytnx::Type.Float || dtype == cytnx::Type.ComplexFloat) {
      tolerance = 1e-5;
    } else {
      tolerance = 1e-10;
    }
    return tolerance;
  }

  ::testing::AssertionResult CheckDivResult(const cytnx::Tensor& gpu_result,
                                            const cytnx::Tensor& left_tensor,
                                            const cytnx::Tensor& right_tensor) {
    // Compare CUDA Div result against CPU Div result
    cytnx::Tensor left_cpu = left_tensor.to(cytnx::Device.cpu);
    cytnx::Tensor right_cpu = right_tensor.to(cytnx::Device.cpu);
    cytnx::Tensor expected_cpu = cytnx::linalg::Div(left_cpu, right_cpu);
    cytnx::Tensor gpu_result_cpu = gpu_result.to(cytnx::Device.cpu);

    cytnx::cytnx_double tolerance = GetTolerance(gpu_result.dtype());

    if (!cytnx::TestTools::AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
      return ::testing::AssertionFailure()
             << "Div result mismatch: CUDA Div result differs from CPU Div result. "
             << "Left dtype: " << left_tensor.dtype() << ", Right dtype: " << right_tensor.dtype()
             << ", tolerance used: " << tolerance;
    }

    return ::testing::AssertionSuccess();
  }

  ::testing::AssertionResult CheckDivScalarResult(const cytnx::Tensor& gpu_result,
                                                  const cytnx::Tensor& tensor,
                                                  const cytnx::cytnx_double& scalar,
                                                  bool scalar_left) {
    // Compare CUDA Div result against CPU Div result
    cytnx::Tensor tensor_cpu = tensor.to(cytnx::Device.cpu);
    cytnx::Tensor expected_cpu;

    if (scalar_left) {
      expected_cpu = cytnx::linalg::Div(scalar, tensor_cpu);
    } else {
      expected_cpu = cytnx::linalg::Div(tensor_cpu, scalar);
    }

    cytnx::Tensor gpu_result_cpu = gpu_result.to(cytnx::Device.cpu);

    cytnx::cytnx_double tolerance = GetTolerance(gpu_result.dtype());

    if (!cytnx::TestTools::AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
      return ::testing::AssertionFailure()
             << "Div scalar result mismatch: CUDA Div result differs from CPU Div result. "
             << "Tensor dtype: " << tensor.dtype() << ", scalar: " << scalar
             << ", scalar_left: " << scalar_left << ", tolerance used: " << tolerance;
    }

    return ::testing::AssertionSuccess();
  }

  std::vector<std::vector<cytnx::cytnx_uint64>> GetTestShapes() {
    std::vector<std::vector<cytnx::cytnx_uint64>> all_shapes;

    auto shapes_1d = cytnx::TestTools::GenerateTestShapes(1, 1, 1024, 4);
    auto shapes_2d = cytnx::TestTools::GenerateTestShapes(2, 1, 512, 4);
    auto shapes_3d = cytnx::TestTools::GenerateTestShapes(3, 1, 64, 4);
    auto shapes_4d = cytnx::TestTools::GenerateTestShapes(4, 1, 32, 4);

    all_shapes.insert(all_shapes.end(), shapes_1d.begin(), shapes_1d.end());
    all_shapes.insert(all_shapes.end(), shapes_2d.begin(), shapes_2d.end());
    all_shapes.insert(all_shapes.end(), shapes_3d.begin(), shapes_3d.end());
    all_shapes.insert(all_shapes.end(), shapes_4d.begin(), shapes_4d.end());

    return all_shapes;
  }

  class DivTestAllShapes : public ::testing::TestWithParam<std::vector<cytnx::cytnx_uint64>> {};

  // Test tensor-to-tensor division
  TEST_P(DivTestAllShapes, gpu_tensor_div_tensor_all_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto dtype : cytnx::TestTools::dtype_list) {
      if (dtype == cytnx::Type.Bool) {
        continue;
      }

      SCOPED_TRACE("Testing Div(tensor, tensor) with shape: " + ::testing::PrintToString(shape) +
                   " and dtype: " + std::to_string(dtype));

      cytnx::Tensor gpu_tensor1 = cytnx::Tensor(shape, dtype, cytnx::Device.cuda);
      cytnx::Tensor gpu_tensor2 = cytnx::Tensor(shape, dtype, cytnx::Device.cuda);
      cytnx::TestTools::InitTensorUniform(gpu_tensor1);
      cytnx::TestTools::InitTensorUniform(gpu_tensor2);
      // Add small offset to avoid division by zero
      gpu_tensor1 = gpu_tensor1 + 1.0;
      gpu_tensor2 = gpu_tensor2 + 1.0;

      cytnx::Tensor gpu_result = cytnx::linalg::Div(gpu_tensor1, gpu_tensor2);
      EXPECT_TRUE(CheckDivResult(gpu_result, gpu_tensor1, gpu_tensor2));

      cytnx::Tensor gpu_result_member = gpu_tensor1.Div(gpu_tensor2);
      EXPECT_TRUE(CheckDivResult(gpu_result_member, gpu_tensor1, gpu_tensor2));

      cytnx::Tensor gpu_result_operator = gpu_tensor1 / gpu_tensor2;
      EXPECT_TRUE(CheckDivResult(gpu_result_operator, gpu_tensor1, gpu_tensor2));
    }
  }

  // Test scalar-to-tensor division
  TEST_P(DivTestAllShapes, gpu_scalar_div_tensor_all_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto dtype : cytnx::TestTools::dtype_list) {
      if (dtype == cytnx::Type.Bool) {
        continue;
      }

      SCOPED_TRACE("Testing Div(scalar, tensor) with shape: " + ::testing::PrintToString(shape) +
                   " and dtype: " + std::to_string(dtype));

      cytnx::Tensor gpu_tensor = cytnx::Tensor(shape, dtype, cytnx::Device.cuda);
      cytnx::TestTools::InitTensorUniform(gpu_tensor);
      // Add small offset to avoid division by zero
      gpu_tensor = gpu_tensor + 1.0;
      cytnx::cytnx_double scalar = 10.0;

      cytnx::Tensor gpu_result = cytnx::linalg::Div(scalar, gpu_tensor);
      EXPECT_TRUE(CheckDivScalarResult(gpu_result, gpu_tensor, scalar, true));

      cytnx::Tensor gpu_result_operator = scalar / gpu_tensor;
      EXPECT_TRUE(CheckDivScalarResult(gpu_result_operator, gpu_tensor, scalar, true));
    }
  }

  // Test tensor-to-scalar division
  TEST_P(DivTestAllShapes, gpu_tensor_div_scalar_all_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto dtype : cytnx::TestTools::dtype_list) {
      if (dtype == cytnx::Type.Bool) {
        continue;
      }

      SCOPED_TRACE("Testing Div(tensor, scalar) with shape: " + ::testing::PrintToString(shape) +
                   " and dtype: " + std::to_string(dtype));

      cytnx::Tensor gpu_tensor = cytnx::Tensor(shape, dtype, cytnx::Device.cuda);
      cytnx::TestTools::InitTensorUniform(gpu_tensor);
      gpu_tensor = gpu_tensor + 1.0;
      cytnx::cytnx_double scalar = 2.0;  // Use non-zero scalar

      cytnx::Tensor gpu_result = cytnx::linalg::Div(gpu_tensor, scalar);
      EXPECT_TRUE(CheckDivScalarResult(gpu_result, gpu_tensor, scalar, false));

      cytnx::Tensor gpu_result_member = gpu_tensor.Div(scalar);
      EXPECT_TRUE(CheckDivScalarResult(gpu_result_member, gpu_tensor, scalar, false));

      cytnx::Tensor gpu_result_operator = gpu_tensor / scalar;
      EXPECT_TRUE(CheckDivScalarResult(gpu_result_operator, gpu_tensor, scalar, false));
    }
  }

  // Test in-place tensor division
  TEST_P(DivTestAllShapes, gpu_tensor_idiv_all_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto dtype : cytnx::TestTools::dtype_list) {
      if (dtype == cytnx::Type.Bool) {
        continue;
      }

      SCOPED_TRACE("Testing iDiv(tensor, tensor) with shape: " + ::testing::PrintToString(shape) +
                   " and dtype: " + std::to_string(dtype));

      cytnx::Tensor gpu_tensor1 = cytnx::Tensor(shape, dtype, cytnx::Device.cuda);
      cytnx::Tensor gpu_tensor2 = cytnx::Tensor(shape, dtype, cytnx::Device.cuda);
      cytnx::TestTools::InitTensorUniform(gpu_tensor1);
      cytnx::TestTools::InitTensorUniform(gpu_tensor2);
      // Add small offset to avoid division by zero
      gpu_tensor1 = gpu_tensor1 + 1.0;
      gpu_tensor2 = gpu_tensor2 + 1.0;

      cytnx::Tensor original_gpu_tensor1 = gpu_tensor1.clone();
      cytnx::Tensor original_gpu_tensor2 = gpu_tensor2.clone();

      cytnx::linalg::iDiv(gpu_tensor1, gpu_tensor2);
      EXPECT_TRUE(CheckDivResult(gpu_tensor1, original_gpu_tensor1, original_gpu_tensor2));

      cytnx::Tensor gpu_tensor1_op = original_gpu_tensor1.clone();
      gpu_tensor1_op /= original_gpu_tensor2;
      EXPECT_TRUE(CheckDivResult(gpu_tensor1_op, original_gpu_tensor1, original_gpu_tensor2));
    }
  }

  INSTANTIATE_TEST_SUITE_P(DivTests, DivTestAllShapes, ::testing::ValuesIn(GetTestShapes()));

}  // namespace DivTest
