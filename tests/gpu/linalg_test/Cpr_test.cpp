#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

namespace cytnx {
  namespace gpu_test {

    ::testing::AssertionResult CheckCprResult(const Tensor& gpu_result, const Tensor& left_tensor,
                                              const Tensor& right_tensor) {
      // Compare CUDA Cpr result against CPU Cpr result
      Tensor left_cpu = left_tensor.to(Device.cpu);
      Tensor right_cpu = right_tensor.to(Device.cpu);
      Tensor expected_cpu = linalg::Cpr(left_cpu, right_cpu);
      Tensor gpu_result_cpu = gpu_result.to(Device.cpu);

      // Cpr returns boolean results, so we need exact comparison
      if (!test::AreEqTensor(gpu_result_cpu, expected_cpu)) {
        return ::testing::AssertionFailure()
               << "Cpr result mismatch: CUDA Cpr result differs from CPU Cpr result. "
               << "Left dtype: " << left_tensor.dtype()
               << ", Right dtype: " << right_tensor.dtype();
      }

      // Check that result is boolean type
      if (gpu_result.dtype() != Type.Bool) {
        return ::testing::AssertionFailure()
               << "Cpr result type mismatch: Expected Bool type but got " << gpu_result.dtype();
      }

      return ::testing::AssertionSuccess();
    }

    ::testing::AssertionResult CheckCprScalarResult(const Tensor& gpu_result, const Tensor& tensor,
                                                    const cytnx_double& scalar, bool scalar_left) {
      // Compare CUDA Cpr result against CPU Cpr result
      Tensor tensor_cpu = tensor.to(Device.cpu);
      Tensor expected_cpu;

      if (scalar_left) {
        expected_cpu = linalg::Cpr(scalar, tensor_cpu);
      } else {
        expected_cpu = linalg::Cpr(tensor_cpu, scalar);
      }

      Tensor gpu_result_cpu = gpu_result.to(Device.cpu);

      // Cpr returns boolean results, so we need exact comparison
      if (!test::AreEqTensor(gpu_result_cpu, expected_cpu)) {
        return ::testing::AssertionFailure()
               << "Cpr scalar result mismatch: CUDA Cpr result differs from CPU Cpr result. "
               << "Tensor dtype: " << tensor.dtype() << ", scalar: " << scalar
               << ", scalar_left: " << scalar_left;
      }

      // Check that result is boolean type
      if (gpu_result.dtype() != Type.Bool) {
        return ::testing::AssertionFailure()
               << "Cpr result type mismatch: Expected Bool type but got " << gpu_result.dtype();
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

    class CprTestAllShapes : public ::testing::TestWithParam<std::vector<cytnx_uint64>> {};

    // Test tensor-to-tensor comparison
    TEST_P(CprTestAllShapes, GpuTensorCprTensorAllTypes) {
      const std::vector<cytnx_uint64>& shape = GetParam();

      for (auto dtype : test::dtype_list) {
        SCOPED_TRACE("Testing Cpr(tensor, tensor) with shape: " + ::testing::PrintToString(shape) +
                     " and dtype: " + std::to_string(dtype));

        Tensor gpu_tensor1 = Tensor(shape, dtype, Device.cuda);
        Tensor gpu_tensor2 = Tensor(shape, dtype, Device.cuda);

        test::InitTensorUniform(gpu_tensor1);
        test::InitTensorUniform(gpu_tensor2);

        Tensor gpu_result = linalg::Cpr(gpu_tensor1, gpu_tensor2);
        EXPECT_TRUE(CheckCprResult(gpu_result, gpu_tensor1, gpu_tensor2));

        Tensor gpu_result_member = gpu_tensor1.Cpr(gpu_tensor2);
        EXPECT_TRUE(CheckCprResult(gpu_result_member, gpu_tensor1, gpu_tensor2));

        Tensor gpu_result_operator = (gpu_tensor1 == gpu_tensor2);
        EXPECT_TRUE(CheckCprResult(gpu_result_operator, gpu_tensor1, gpu_tensor2));
      }
    }

    // Test scalar-to-tensor comparison
    TEST_P(CprTestAllShapes, GpuScalarCprTensorAllTypes) {
      const std::vector<cytnx_uint64>& shape = GetParam();

      for (auto dtype : test::dtype_list) {
        SCOPED_TRACE("Testing Cpr(scalar, tensor) with shape: " + ::testing::PrintToString(shape) +
                     " and dtype: " + std::to_string(dtype));

        Tensor gpu_tensor = Tensor(shape, dtype, Device.cuda);
        test::InitTensorUniform(gpu_tensor);

        const cytnx_double scalar = 2.3;

        Tensor gpu_result = linalg::Cpr(scalar, gpu_tensor);
        EXPECT_TRUE(CheckCprScalarResult(gpu_result, gpu_tensor, scalar, true));

        Tensor gpu_result_operator = (scalar == gpu_tensor);
        EXPECT_TRUE(CheckCprScalarResult(gpu_result_operator, gpu_tensor, scalar, true));
      }
    }

    // Test tensor-to-scalar comparison
    TEST_P(CprTestAllShapes, GpuTensorCprScalarAllTypes) {
      const std::vector<cytnx_uint64>& shape = GetParam();

      for (auto dtype : test::dtype_list) {
        SCOPED_TRACE("Testing Cpr(tensor, scalar) with shape: " + ::testing::PrintToString(shape) +
                     " and dtype: " + std::to_string(dtype));

        Tensor gpu_tensor = Tensor(shape, dtype, Device.cuda);
        test::InitTensorUniform(gpu_tensor);

        const cytnx_double scalar = 2.3;

        Tensor gpu_result = linalg::Cpr(gpu_tensor, scalar);
        EXPECT_TRUE(CheckCprScalarResult(gpu_result, gpu_tensor, scalar, false));

        Tensor gpu_result_member = gpu_tensor.Cpr(scalar);
        EXPECT_TRUE(CheckCprScalarResult(gpu_result_member, gpu_tensor, scalar, false));

        Tensor gpu_result_operator = (gpu_tensor == scalar);
        EXPECT_TRUE(CheckCprScalarResult(gpu_result_operator, gpu_tensor, scalar, false));
      }
    }

    // Test comparison with same tensors (all elements should be true)
    TEST_P(CprTestAllShapes, GpuTensorCprIdentical) {
      const std::vector<cytnx_uint64>& shape = GetParam();

      for (auto dtype : test::dtype_list) {
        SCOPED_TRACE("Testing Cpr with identical tensors, shape: " +
                     ::testing::PrintToString(shape) + " and dtype: " + std::to_string(dtype));

        Tensor gpu_tensor = Tensor(shape, dtype, Device.cuda);
        test::InitTensorUniform(gpu_tensor);

        // Compare tensor with itself
        Tensor gpu_result = linalg::Cpr(gpu_tensor, gpu_tensor);

        // Result should be all true
        Tensor expected = ones(shape, Type.Bool, Device.cuda);
        Tensor gpu_result_cpu = gpu_result.to(Device.cpu);
        Tensor expected_cpu = expected.to(Device.cpu);

        EXPECT_TRUE(test::AreEqTensor(gpu_result_cpu, expected_cpu))
          << "Comparison of identical tensors should yield all true values";
      }
    }

    // Test comparison with totally different tensors
    TEST_P(CprTestAllShapes, GpuTensorCprDifferent) {
      const std::vector<cytnx_uint64>& shape = GetParam();

      for (auto dtype : test::dtype_list) {
        SCOPED_TRACE("Testing Cpr with different tensors, shape: " +
                     ::testing::PrintToString(shape) + " and dtype: " + std::to_string(dtype));

        Tensor gpu_tensor1 = Tensor(shape, dtype, Device.cuda);
        Tensor gpu_tensor2 = Tensor(shape, dtype, Device.cuda);

        // Initialize both tensors with the same value first
        test::InitTensorUniform(gpu_tensor1);
        gpu_tensor2 = gpu_tensor1.clone();

        // Make tensor2 different
        if (dtype == Type.Bool) {
          // For Bool type, flip all values (0->1, 1->0)
          gpu_tensor2 = ones(shape, Type.Bool, Device.cuda) - gpu_tensor2;
        } else if (dtype == Type.ComplexDouble || dtype == Type.ComplexFloat) {
          gpu_tensor2 = gpu_tensor2 + cytnx_complex128(100.0, 100.0);
        } else {
          gpu_tensor2 = gpu_tensor2 + 100.0;
        }

        Tensor gpu_result = linalg::Cpr(gpu_tensor1, gpu_tensor2);

        // Result should be all false
        Tensor expected = zeros(shape, Type.Bool, Device.cuda);
        Tensor gpu_result_cpu = gpu_result.to(Device.cpu);
        Tensor expected_cpu = expected.to(Device.cpu);

        EXPECT_TRUE(test::AreEqTensor(gpu_result_cpu, expected_cpu))
          << "Comparison of completely different tensors should yield all false values";
      }
    }

    INSTANTIATE_TEST_SUITE_P(CprTests, CprTestAllShapes, ::testing::ValuesIn(GetTestShapes()));

  }  // namespace gpu_test
}  // namespace cytnx
