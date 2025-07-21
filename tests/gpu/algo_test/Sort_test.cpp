#include "gtest/gtest.h"

#include "gpu_test_tools.h"
#include "cytnx.hpp"

namespace SortTest {

  ::testing::AssertionResult CheckResult(const cytnx::Tensor& sorted,
                                         const cytnx::Tensor& original);

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

  class SortTestAllShapes : public ::testing::TestWithParam<std::vector<cytnx::cytnx_uint64>> {};

  TEST_P(SortTestAllShapes, gpu_tensor_all_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto dtype : cytnx::TestTools::dtype_list) {
      if (dtype == cytnx::Type.Bool) {
        continue;
      }
      SCOPED_TRACE("Testing with shape: " + ::testing::PrintToString(shape) +
                   " and dtype: " + std::to_string(dtype));
      cytnx::Tensor in = cytnx::Tensor(shape, dtype).to(cytnx::Device.cuda);
      cytnx::TestTools::InitTensorUniform(in);
      cytnx::Tensor sorted = cytnx::algo::Sort(in);
      EXPECT_TRUE(CheckResult(sorted, in));
    }
  }

  INSTANTIATE_TEST_SUITE_P(SortTests, SortTestAllShapes, ::testing::ValuesIn(GetTestShapes()));

  ::testing::AssertionResult CheckResult(const cytnx::Tensor& sorted,
                                         const cytnx::Tensor& original) {
    // Compare CUDA sort result and CPU sort result
    cytnx::Tensor original_cpu = original.to(cytnx::Device.cpu);
    cytnx::Tensor expected = cytnx::algo::Sort(original_cpu);
    cytnx::Tensor sorted_cpu = sorted.to(cytnx::Device.cpu);

    if (!cytnx::TestTools::AreEqTensor(sorted_cpu, expected)) {
      return ::testing::AssertionFailure()
             << "Sort result mismatch: CUDA sort result differs from CPU sort result";
    }

    return ::testing::AssertionSuccess();
  }

}  // namespace SortTest
