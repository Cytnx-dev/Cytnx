#include "gtest/gtest.h"

#include "gpu_test_tools.h"
#include "cytnx.hpp"

namespace CprTest {

  ::testing::AssertionResult CheckCprResult(const cytnx::Tensor& gpu_result,
                                            const cytnx::Tensor& left_tensor,
                                            const cytnx::Tensor& right_tensor) {
    // Compare CUDA Cpr result against CPU Cpr result
    cytnx::Tensor left_cpu = left_tensor.to(cytnx::Device.cpu);
    cytnx::Tensor right_cpu = right_tensor.to(cytnx::Device.cpu);
    cytnx::Tensor expected_cpu = cytnx::linalg::Cpr(left_cpu, right_cpu);
    cytnx::Tensor gpu_result_cpu = gpu_result.to(cytnx::Device.cpu);

    // Cpr returns boolean results, so we need exact comparison
    if (!cytnx::TestTools::AreEqTensor(gpu_result_cpu, expected_cpu)) {
      return ::testing::AssertionFailure()
             << "Cpr result mismatch: CUDA Cpr result differs from CPU Cpr result. "
             << "Left dtype: " << left_tensor.dtype() << ", Right dtype: " << right_tensor.dtype();
    }

    // Check that result is boolean type
    if (gpu_result.dtype() != cytnx::Type.Bool) {
      return ::testing::AssertionFailure()
             << "Cpr result type mismatch: Expected Bool type but got " << gpu_result.dtype();
    }

    return ::testing::AssertionSuccess();
  }

  ::testing::AssertionResult CheckCprScalarResult(const cytnx::Tensor& gpu_result,
                                                  const cytnx::Tensor& tensor,
                                                  const cytnx::cytnx_double& scalar,
                                                  bool scalar_left) {
    // Compare CUDA Cpr result against CPU Cpr result
    cytnx::Tensor tensor_cpu = tensor.to(cytnx::Device.cpu);
    cytnx::Tensor expected_cpu;

    if (scalar_left) {
      expected_cpu = cytnx::linalg::Cpr(scalar, tensor_cpu);
    } else {
      expected_cpu = cytnx::linalg::Cpr(tensor_cpu, scalar);
    }

    cytnx::Tensor gpu_result_cpu = gpu_result.to(cytnx::Device.cpu);

    // Cpr returns boolean results, so we need exact comparison
    if (!cytnx::TestTools::AreEqTensor(gpu_result_cpu, expected_cpu)) {
      return ::testing::AssertionFailure()
             << "Cpr scalar result mismatch: CUDA Cpr result differs from CPU Cpr result. "
             << "Tensor dtype: " << tensor.dtype() << ", scalar: " << scalar
             << ", scalar_left: " << scalar_left;
    }

    // Check that result is boolean type
    if (gpu_result.dtype() != cytnx::Type.Bool) {
      return ::testing::AssertionFailure()
             << "Cpr result type mismatch: Expected Bool type but got " << gpu_result.dtype();
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

  class CprTestAllShapes : public ::testing::TestWithParam<std::vector<cytnx::cytnx_uint64>> {};

  // Test tensor-to-tensor comparison
  TEST_P(CprTestAllShapes, gpu_tensor_cpr_tensor_all_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto dtype : cytnx::TestTools::dtype_list) {
      SCOPED_TRACE("Testing Cpr(tensor, tensor) with shape: " + ::testing::PrintToString(shape) +
                   " and dtype: " + std::to_string(dtype));

      cytnx::Tensor gpu_tensor1 = cytnx::Tensor(shape, dtype, cytnx::Device.cuda);
      cytnx::Tensor gpu_tensor2 = cytnx::Tensor(shape, dtype, cytnx::Device.cuda);

      cytnx::TestTools::InitTensorUniform(gpu_tensor1);
      cytnx::TestTools::InitTensorUniform(gpu_tensor2);

      cytnx::Tensor gpu_result = cytnx::linalg::Cpr(gpu_tensor1, gpu_tensor2);
      EXPECT_TRUE(CheckCprResult(gpu_result, gpu_tensor1, gpu_tensor2));

      cytnx::Tensor gpu_result_member = gpu_tensor1.Cpr(gpu_tensor2);
      EXPECT_TRUE(CheckCprResult(gpu_result_member, gpu_tensor1, gpu_tensor2));

      cytnx::Tensor gpu_result_operator = (gpu_tensor1 == gpu_tensor2);
      EXPECT_TRUE(CheckCprResult(gpu_result_operator, gpu_tensor1, gpu_tensor2));
    }
  }

  // Test scalar-to-tensor comparison
  TEST_P(CprTestAllShapes, gpu_scalar_cpr_tensor_all_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto dtype : cytnx::TestTools::dtype_list) {
      SCOPED_TRACE("Testing Cpr(scalar, tensor) with shape: " + ::testing::PrintToString(shape) +
                   " and dtype: " + std::to_string(dtype));

      cytnx::Tensor gpu_tensor = cytnx::Tensor(shape, dtype, cytnx::Device.cuda);
      cytnx::TestTools::InitTensorUniform(gpu_tensor);

      const cytnx::cytnx_double scalar = 2.3;

      cytnx::Tensor gpu_result = cytnx::linalg::Cpr(scalar, gpu_tensor);
      EXPECT_TRUE(CheckCprScalarResult(gpu_result, gpu_tensor, scalar, true));

      cytnx::Tensor gpu_result_operator = (scalar == gpu_tensor);
      EXPECT_TRUE(CheckCprScalarResult(gpu_result_operator, gpu_tensor, scalar, true));
    }
  }

  // Test tensor-to-scalar comparison
  TEST_P(CprTestAllShapes, gpu_tensor_cpr_scalar_all_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto dtype : cytnx::TestTools::dtype_list) {
      SCOPED_TRACE("Testing Cpr(tensor, scalar) with shape: " + ::testing::PrintToString(shape) +
                   " and dtype: " + std::to_string(dtype));

      cytnx::Tensor gpu_tensor = cytnx::Tensor(shape, dtype, cytnx::Device.cuda);
      cytnx::TestTools::InitTensorUniform(gpu_tensor);

      const cytnx::cytnx_double scalar = 2.3;

      cytnx::Tensor gpu_result = cytnx::linalg::Cpr(gpu_tensor, scalar);
      EXPECT_TRUE(CheckCprScalarResult(gpu_result, gpu_tensor, scalar, false));

      cytnx::Tensor gpu_result_member = gpu_tensor.Cpr(scalar);
      EXPECT_TRUE(CheckCprScalarResult(gpu_result_member, gpu_tensor, scalar, false));

      cytnx::Tensor gpu_result_operator = (gpu_tensor == scalar);
      EXPECT_TRUE(CheckCprScalarResult(gpu_result_operator, gpu_tensor, scalar, false));
    }
  }

  // Test comparison with same tensors (all elements should be true)
  TEST_P(CprTestAllShapes, gpu_tensor_cpr_identical) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto dtype : cytnx::TestTools::dtype_list) {
      SCOPED_TRACE("Testing Cpr with identical tensors, shape: " + ::testing::PrintToString(shape) +
                   " and dtype: " + std::to_string(dtype));

      cytnx::Tensor gpu_tensor = cytnx::Tensor(shape, dtype, cytnx::Device.cuda);
      cytnx::TestTools::InitTensorUniform(gpu_tensor);

      // Compare tensor with itself
      cytnx::Tensor gpu_result = cytnx::linalg::Cpr(gpu_tensor, gpu_tensor);

      // Result should be all true
      cytnx::Tensor expected = cytnx::ones(shape, cytnx::Type.Bool, cytnx::Device.cuda);
      cytnx::Tensor gpu_result_cpu = gpu_result.to(cytnx::Device.cpu);
      cytnx::Tensor expected_cpu = expected.to(cytnx::Device.cpu);

      EXPECT_TRUE(cytnx::TestTools::AreEqTensor(gpu_result_cpu, expected_cpu))
        << "Comparison of identical tensors should yield all true values";
    }
  }

  // Test comparison with totally different tensors
  TEST_P(CprTestAllShapes, gpu_tensor_cpr_different) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto dtype : cytnx::TestTools::dtype_list) {
      SCOPED_TRACE("Testing Cpr with different tensors, shape: " + ::testing::PrintToString(shape) +
                   " and dtype: " + std::to_string(dtype));

      cytnx::Tensor gpu_tensor1 = cytnx::Tensor(shape, dtype, cytnx::Device.cuda);
      cytnx::Tensor gpu_tensor2 = cytnx::Tensor(shape, dtype, cytnx::Device.cuda);

      // Initialize both tensors with the same value first
      cytnx::TestTools::InitTensorUniform(gpu_tensor1);
      gpu_tensor2 = gpu_tensor1.clone();

      // Make tensor2 different
      if (dtype == cytnx::Type.Bool) {
        // For Bool type, flip all values (0->1, 1->0)
        gpu_tensor2 = cytnx::ones(shape, cytnx::Type.Bool, cytnx::Device.cuda) - gpu_tensor2;
      } else if (dtype == cytnx::Type.ComplexDouble || dtype == cytnx::Type.ComplexFloat) {
        gpu_tensor2 = gpu_tensor2 + cytnx::cytnx_complex128(100.0, 100.0);
      } else {
        gpu_tensor2 = gpu_tensor2 + 100.0;
      }

      cytnx::Tensor gpu_result = cytnx::linalg::Cpr(gpu_tensor1, gpu_tensor2);

      // Result should be all false
      cytnx::Tensor expected = cytnx::zeros(shape, cytnx::Type.Bool, cytnx::Device.cuda);
      cytnx::Tensor gpu_result_cpu = gpu_result.to(cytnx::Device.cpu);
      cytnx::Tensor expected_cpu = expected.to(cytnx::Device.cpu);

      EXPECT_TRUE(cytnx::TestTools::AreEqTensor(gpu_result_cpu, expected_cpu))
        << "Comparison of completely different tensors should yield all false values";
    }
  }

  INSTANTIATE_TEST_SUITE_P(CprTests, CprTestAllShapes, ::testing::ValuesIn(GetTestShapes()));

}  // namespace CprTest
