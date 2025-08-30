#include "gtest/gtest.h"

#include "gpu_test_tools.h"
#include "cytnx.hpp"

namespace MulTest {

  ::testing::AssertionResult CheckMulResult(const cytnx::Tensor& gpu_result,
                                            const cytnx::Tensor& left_tensor,
                                            const cytnx::Tensor& right_tensor);

  ::testing::AssertionResult CheckMulScalarResult(const cytnx::Tensor& gpu_result,
                                                  const cytnx::Tensor& tensor,
                                                  const cytnx::cytnx_double& scalar,
                                                  bool scalar_left = false);

  std::vector<std::vector<cytnx::cytnx_uint64>> GetTestShapes();

  cytnx::cytnx_double GetTolerance(const unsigned int& dtype);

  class MulTestAllShapes : public ::testing::TestWithParam<std::vector<cytnx::cytnx_uint64>> {};

  // Test tensor-to-tensor multiplication
  TEST_P(MulTestAllShapes, gpu_tensor_mul_tensor_all_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto dtype : cytnx::TestTools::dtype_list) {
      if (dtype == cytnx::Type.Bool) {
        continue;
      }

      SCOPED_TRACE("Testing Mul(tensor, tensor) with shape: " + ::testing::PrintToString(shape) +
                   " and dtype: " + std::to_string(dtype));

      cytnx::Tensor gpu_tensor1 = cytnx::Tensor(shape, dtype).to(cytnx::Device.cuda);
      cytnx::Tensor gpu_tensor2 = cytnx::Tensor(shape, dtype).to(cytnx::Device.cuda);
      cytnx::TestTools::InitTensorUniform(gpu_tensor1);
      cytnx::TestTools::InitTensorUniform(gpu_tensor2);

      cytnx::Tensor gpu_result = cytnx::linalg::Mul(gpu_tensor1, gpu_tensor2);
      EXPECT_TRUE(CheckMulResult(gpu_result, gpu_tensor1, gpu_tensor2));

      cytnx::Tensor gpu_result_member = gpu_tensor1.Mul(gpu_tensor2);
      EXPECT_TRUE(CheckMulResult(gpu_result_member, gpu_tensor1, gpu_tensor2));

      cytnx::Tensor gpu_result_operator = gpu_tensor1 * gpu_tensor2;
      EXPECT_TRUE(CheckMulResult(gpu_result_operator, gpu_tensor1, gpu_tensor2));
    }
  }

  // Test scalar-to-tensor multiplication
  TEST_P(MulTestAllShapes, gpu_scalar_mul_tensor_all_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto dtype : cytnx::TestTools::dtype_list) {
      if (dtype == cytnx::Type.Bool) {
        continue;
      }

      SCOPED_TRACE("Testing Mul(scalar, tensor) with shape: " + ::testing::PrintToString(shape) +
                   " and dtype: " + std::to_string(dtype));

      cytnx::Tensor gpu_tensor = cytnx::Tensor(shape, dtype).to(cytnx::Device.cuda);
      cytnx::TestTools::InitTensorUniform(gpu_tensor);
      cytnx::cytnx_double scalar = 2.3;

      cytnx::Tensor gpu_result = cytnx::linalg::Mul(scalar, gpu_tensor);
      EXPECT_TRUE(CheckMulScalarResult(gpu_result, gpu_tensor, scalar, true));

      cytnx::Tensor gpu_result_operator = scalar * gpu_tensor;
      EXPECT_TRUE(CheckMulScalarResult(gpu_result_operator, gpu_tensor, scalar, true));
    }
  }

  // Test tensor-to-scalar multiplication
  TEST_P(MulTestAllShapes, gpu_tensor_mul_scalar_all_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto dtype : cytnx::TestTools::dtype_list) {
      if (dtype == cytnx::Type.Bool) {
        continue;
      }

      SCOPED_TRACE("Testing Mul(tensor, scalar) with shape: " + ::testing::PrintToString(shape) +
                   " and dtype: " + std::to_string(dtype));

      cytnx::Tensor gpu_tensor = cytnx::Tensor(shape, dtype).to(cytnx::Device.cuda);
      cytnx::TestTools::InitTensorUniform(gpu_tensor);
      cytnx::cytnx_double scalar = 2.3;

      cytnx::Tensor gpu_result = cytnx::linalg::Mul(gpu_tensor, scalar);
      EXPECT_TRUE(CheckMulScalarResult(gpu_result, gpu_tensor, scalar, false));

      cytnx::Tensor gpu_result_member = gpu_tensor.Mul(scalar);
      EXPECT_TRUE(CheckMulScalarResult(gpu_result_member, gpu_tensor, scalar, false));

      cytnx::Tensor gpu_result_operator = gpu_tensor * scalar;
      EXPECT_TRUE(CheckMulScalarResult(gpu_result_operator, gpu_tensor, scalar, false));
    }
  }

  // Test in-place tensor multiplication
  TEST_P(MulTestAllShapes, gpu_tensor_imul_all_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto dtype : cytnx::TestTools::dtype_list) {
      if (dtype == cytnx::Type.Bool) {
        continue;
      }

      SCOPED_TRACE("Testing iMul(tensor, tensor) with shape: " + ::testing::PrintToString(shape) +
                   " and dtype: " + std::to_string(dtype));

      cytnx::Tensor gpu_tensor1 = cytnx::Tensor(shape, dtype).to(cytnx::Device.cuda);
      cytnx::Tensor gpu_tensor2 = cytnx::Tensor(shape, dtype).to(cytnx::Device.cuda);
      cytnx::TestTools::InitTensorUniform(gpu_tensor1);
      cytnx::TestTools::InitTensorUniform(gpu_tensor2);

      cytnx::Tensor original_gpu_tensor1 = gpu_tensor1.clone();
      cytnx::Tensor original_gpu_tensor2 = gpu_tensor2.clone();

      cytnx::linalg::iMul(gpu_tensor1, gpu_tensor2);
      EXPECT_TRUE(CheckMulResult(gpu_tensor1, original_gpu_tensor1, original_gpu_tensor2));

      cytnx::Tensor gpu_tensor1_op = original_gpu_tensor1.clone();
      gpu_tensor1_op *= original_gpu_tensor2;
      EXPECT_TRUE(CheckMulResult(gpu_tensor1_op, original_gpu_tensor1, original_gpu_tensor2));
    }
  }

  INSTANTIATE_TEST_SUITE_P(MulTests, MulTestAllShapes, ::testing::ValuesIn(GetTestShapes()));

  ::testing::AssertionResult CheckMulResult(const cytnx::Tensor& gpu_result,
                                            const cytnx::Tensor& left_tensor,
                                            const cytnx::Tensor& right_tensor) {
    // Compare CUDA Mul result against CPU Mul result
    cytnx::Tensor left_cpu = left_tensor.to(cytnx::Device.cpu);
    cytnx::Tensor right_cpu = right_tensor.to(cytnx::Device.cpu);
    cytnx::Tensor expected_cpu = cytnx::linalg::Mul(left_cpu, right_cpu);
    cytnx::Tensor gpu_result_cpu = gpu_result.to(cytnx::Device.cpu);

    cytnx::cytnx_double tolerance = GetTolerance(gpu_result.dtype());

    if (!cytnx::TestTools::AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
      return ::testing::AssertionFailure()
             << "Mul result mismatch: CUDA Mul result differs from CPU Mul result. "
             << "Left dtype: " << left_tensor.dtype() << ", Right dtype: " << right_tensor.dtype()
             << ", tolerance used: " << tolerance;
    }

    return ::testing::AssertionSuccess();
  }

  ::testing::AssertionResult CheckMulScalarResult(const cytnx::Tensor& gpu_result,
                                                  const cytnx::Tensor& tensor,
                                                  const cytnx::cytnx_double& scalar,
                                                  bool scalar_left) {
    // Compare CUDA Mul result against CPU Mul result
    cytnx::Tensor tensor_cpu = tensor.to(cytnx::Device.cpu);
    cytnx::Tensor expected_cpu;

    if (scalar_left) {
      expected_cpu = cytnx::linalg::Mul(scalar, tensor_cpu);
    } else {
      expected_cpu = cytnx::linalg::Mul(tensor_cpu, scalar);
    }

    cytnx::Tensor gpu_result_cpu = gpu_result.to(cytnx::Device.cpu);

    cytnx::cytnx_double tolerance = GetTolerance(gpu_result.dtype());

    if (!cytnx::TestTools::AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
      return ::testing::AssertionFailure()
             << "Mul scalar result mismatch: CUDA Mul result differs from CPU Mul result. "
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

  cytnx::cytnx_double GetTolerance(const unsigned int& dtype) {
    cytnx::cytnx_double tolerance = 1e-6;
    if (dtype == cytnx::Type.ComplexFloat) {
      tolerance = 0.1;
    }
    return tolerance;
  }

}  // namespace MulTest
