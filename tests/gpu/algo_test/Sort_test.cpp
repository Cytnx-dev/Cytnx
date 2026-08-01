#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

namespace cytnx {
  namespace gpu_test {
    namespace {

      ::testing::AssertionResult CheckResult(const Tensor& sorted, const Tensor& original);

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

      class SortTestAllShapes : public ::testing::TestWithParam<std::vector<cytnx_uint64>> {};

      TEST_P(SortTestAllShapes, GpuTensorAllTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto dtype : dtype_list) {
          if (dtype == Type.Bool) {
            continue;
          }
          SCOPED_TRACE("Testing with shape: " + ::testing::PrintToString(shape) +
                       " and dtype: " + std::to_string(dtype));
          Tensor in = Tensor(shape, dtype).to(Device.cuda);
          InitTensorUniform(in);
          Tensor sorted = algo::Sort(in);
          EXPECT_TRUE(CheckResult(sorted, in));
        }
      }

      INSTANTIATE_TEST_SUITE_P(SortTests, SortTestAllShapes, ::testing::ValuesIn(GetTestShapes()));

      ::testing::AssertionResult CheckResult(const Tensor& sorted, const Tensor& original) {
        // Compare CUDA sort result and CPU sort result
        Tensor original_cpu = original.to(Device.cpu);
        Tensor expected = algo::Sort(original_cpu);
        Tensor sorted_cpu = sorted.to(Device.cpu);

        if (!AreEqTensor(sorted_cpu, expected)) {
          return ::testing::AssertionFailure()
                 << "Sort result mismatch: CUDA sort result differs from CPU sort result";
        }

        return ::testing::AssertionSuccess();
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
