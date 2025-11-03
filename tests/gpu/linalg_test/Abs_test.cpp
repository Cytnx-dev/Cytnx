#include "gtest/gtest.h"

#include "gpu_test_tools.h"
#include "cytnx.hpp"

namespace AbsTest {

  ::testing::AssertionResult CheckAbsResult(const cytnx::Tensor& gpu_result,
                                            const cytnx::Tensor& original_gpu_tensor);

  std::vector<std::vector<cytnx::cytnx_uint64>> GetTestShapes();

  class AbsTestAllShapes : public ::testing::TestWithParam<std::vector<cytnx::cytnx_uint64>> {};

  TEST_P(AbsTestAllShapes, gpu_tensor_abs_all_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto dtype : cytnx::TestTools::dtype_list) {
      if (dtype == cytnx::Type.Bool) {
        continue;
      }

      SCOPED_TRACE("Testing Abs with shape: " + ::testing::PrintToString(shape) +
                   " and dtype: " + std::to_string(dtype));

      cytnx::Tensor gpu_tensor = cytnx::Tensor(shape, dtype).to(cytnx::Device.cuda);
      cytnx::TestTools::InitTensorUniform(gpu_tensor);

      // Test standalone function
      cytnx::Tensor gpu_result = cytnx::linalg::Abs(gpu_tensor);
      EXPECT_TRUE(CheckAbsResult(gpu_result, gpu_tensor));

      // Test member function
      cytnx::Tensor gpu_result_member = gpu_tensor.Abs();
      EXPECT_TRUE(CheckAbsResult(gpu_result_member, gpu_tensor));
    }
  }

  TEST_P(AbsTestAllShapes, gpu_tensor_abs_inplace_all_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto dtype : cytnx::TestTools::dtype_list) {
      if (dtype == cytnx::Type.Bool) {
        continue;
      }
      SCOPED_TRACE("Testing Abs_ with shape: " + ::testing::PrintToString(shape) +
                   " and dtype: " + std::to_string(dtype));

      cytnx::Tensor gpu_tensor = cytnx::Tensor(shape, dtype).to(cytnx::Device.cuda);
      cytnx::TestTools::InitTensorUniform(gpu_tensor);
      cytnx::Tensor original_copy = gpu_tensor.clone();

      // Test standalone in-place function
      cytnx::linalg::Abs_(gpu_tensor);
      EXPECT_TRUE(CheckAbsResult(gpu_tensor, original_copy));

      // Test member in-place function
      cytnx::Tensor gpu_tensor_member = original_copy.clone();
      gpu_tensor_member.Abs_();
      EXPECT_TRUE(CheckAbsResult(gpu_tensor_member, original_copy));
    }
  }

  INSTANTIATE_TEST_SUITE_P(AbsTests, AbsTestAllShapes, ::testing::ValuesIn(GetTestShapes()));

  ::testing::AssertionResult CheckAbsResult(const cytnx::Tensor& gpu_result,
                                            const cytnx::Tensor& original_gpu_tensor) {
    // Compare CUDA Abs result against CPU Abs result
    cytnx::Tensor original_cpu = original_gpu_tensor.to(cytnx::Device.cpu);
    cytnx::Tensor expected_cpu = cytnx::linalg::Abs(original_cpu);
    cytnx::Tensor gpu_result_cpu = gpu_result.to(cytnx::Device.cpu);

    cytnx::cytnx_double tolerance = 1e-6;
    if (original_gpu_tensor.dtype() == cytnx::Type.ComplexFloat) {
      tolerance = 1e-3;
    }

    if (!cytnx::TestTools::AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
      return ::testing::AssertionFailure()
             << "Abs result mismatch: CUDA Abs result differs from CPU Abs result. "
             << "Original dtype: " << original_gpu_tensor.dtype()
             << ", tolerance used: " << tolerance;
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

}  // namespace AbsTest
