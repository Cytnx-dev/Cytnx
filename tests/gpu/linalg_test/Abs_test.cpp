#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

namespace cytnx {
  namespace gpu_test {
    namespace {

      ::testing::AssertionResult CheckAbsResult(const Tensor& gpu_result,
                                                const Tensor& original_gpu_tensor);

      std::vector<std::vector<cytnx_uint64>> GetTestShapes();

      class AbsTestAllShapes : public ::testing::TestWithParam<std::vector<cytnx_uint64>> {};

      TEST_P(AbsTestAllShapes, GpuTensorAbsAllTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto dtype : test::dtype_list) {
          if (dtype == Type.Bool) {
            continue;
          }

          SCOPED_TRACE("Testing Abs with shape: " + ::testing::PrintToString(shape) +
                       " and dtype: " + std::to_string(dtype));

          Tensor gpu_tensor = Tensor(shape, dtype).to(Device.cuda);
          test::InitTensorUniform(gpu_tensor);

          // Test standalone function
          Tensor gpu_result = linalg::Abs(gpu_tensor);
          EXPECT_TRUE(CheckAbsResult(gpu_result, gpu_tensor));

          // Test member function
          Tensor gpu_result_member = gpu_tensor.Abs();
          EXPECT_TRUE(CheckAbsResult(gpu_result_member, gpu_tensor));
        }
      }

      TEST_P(AbsTestAllShapes, GpuTensorAbsInplaceAllTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto dtype : test::dtype_list) {
          if (dtype == Type.Bool) {
            continue;
          }
          SCOPED_TRACE("Testing Abs_ with shape: " + ::testing::PrintToString(shape) +
                       " and dtype: " + std::to_string(dtype));

          Tensor gpu_tensor = Tensor(shape, dtype).to(Device.cuda);
          test::InitTensorUniform(gpu_tensor);
          Tensor original_copy = gpu_tensor.clone();

          // Test standalone in-place function
          linalg::Abs_(gpu_tensor);
          EXPECT_TRUE(CheckAbsResult(gpu_tensor, original_copy));

          // Test member in-place function
          Tensor gpu_tensor_member = original_copy.clone();
          gpu_tensor_member.Abs_();
          EXPECT_TRUE(CheckAbsResult(gpu_tensor_member, original_copy));
        }
      }

      INSTANTIATE_TEST_SUITE_P(AbsTests, AbsTestAllShapes, ::testing::ValuesIn(GetTestShapes()));

      ::testing::AssertionResult CheckAbsResult(const Tensor& gpu_result,
                                                const Tensor& original_gpu_tensor) {
        // Compare CUDA Abs result against CPU Abs result
        Tensor original_cpu = original_gpu_tensor.to(Device.cpu);
        Tensor expected_cpu = linalg::Abs(original_cpu);
        Tensor gpu_result_cpu = gpu_result.to(Device.cpu);

        cytnx_double tolerance = 1e-6;
        if (original_gpu_tensor.dtype() == Type.ComplexFloat) {
          tolerance = 1e-3;
        }

        if (!test::AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
          return ::testing::AssertionFailure()
                 << "Abs result mismatch: CUDA Abs result differs from CPU Abs result. "
                 << "Original dtype: " << original_gpu_tensor.dtype()
                 << ", tolerance used: " << tolerance;
        }

        return ::testing::AssertionSuccess();
      }

      std::vector<std::vector<cytnx_uint64>> GetTestShapes() {
        std::vector<std::vector<cytnx_uint64>> all_shapes;

        auto shapes_1d = test::GenerateTestShapes(1, 1, 1024, 4);
        auto shapes_2d = test::GenerateTestShapes(2, 1, 512, 4);
        auto shapes_3d = test::GenerateTestShapes(3, 1, 64, 4);
        auto shapes_4d = test::GenerateTestShapes(4, 1, 32, 4);

        all_shapes.insert(all_shapes.end(), shapes_1d.begin(), shapes_1d.end());
        all_shapes.insert(all_shapes.end(), shapes_2d.begin(), shapes_2d.end());
        all_shapes.insert(all_shapes.end(), shapes_3d.begin(), shapes_3d.end());
        all_shapes.insert(all_shapes.end(), shapes_4d.begin(), shapes_4d.end());

        return all_shapes;
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
