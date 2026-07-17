#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

namespace cytnx {
  namespace gpu_test {

    static cytnx_double GetTolerance(const unsigned int& dtype) {
      cytnx_double tolerance;
      if (dtype == Type.Float) {
        tolerance = 1e-4;
      } else {
        tolerance = 1e-5;
      }
      return tolerance;
    }

    ::testing::AssertionResult CheckModResult(const Tensor& gpu_result, const Tensor& left_tensor,
                                              const Tensor& right_tensor) {
      // Compare CUDA Mod result against CPU Mod result
      Tensor left_cpu = left_tensor.to(Device.cpu);
      Tensor right_cpu = right_tensor.to(Device.cpu);
      Tensor expected_cpu = linalg::Mod(left_cpu, right_cpu);
      Tensor gpu_result_cpu = gpu_result.to(Device.cpu);
      cytnx_double tolerance = GetTolerance(gpu_result.dtype());

      if (!test::AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
        return ::testing::AssertionFailure()
               << "Mod result mismatch: CUDA Mod result differs from CPU Mod result. "
               << "Left dtype: " << left_tensor.dtype()
               << ", Right dtype: " << right_tensor.dtype();
      }

      // Check that result type matches expected output type
      if (gpu_result.dtype() != expected_cpu.dtype()) {
        return ::testing::AssertionFailure()
               << "Mod result type mismatch: Expected " << expected_cpu.dtype() << " but got "
               << gpu_result.dtype();
      }

      return ::testing::AssertionSuccess();
    }

    ::testing::AssertionResult CheckModScalarResult(const Tensor& gpu_result, const Tensor& tensor,
                                                    const cytnx_double& scalar, bool scalar_left) {
      // Compare CUDA Mod result against CPU Mod result
      Tensor tensor_cpu = tensor.to(Device.cpu);
      Tensor expected_cpu;

      if (scalar_left) {
        expected_cpu = linalg::Mod(scalar, tensor_cpu);
      } else {
        expected_cpu = linalg::Mod(tensor_cpu, scalar);
      }

      Tensor gpu_result_cpu = gpu_result.to(Device.cpu);
      cytnx_double tolerance = GetTolerance(tensor.dtype());
      std::cout << tolerance << std::endl;

      if (!test::AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
        return ::testing::AssertionFailure()
               << "Mod scalar result mismatch: CUDA Mod result differs from CPU Mod result. "
               << "Tensor dtype: " << tensor.dtype() << ", scalar: " << scalar
               << ", scalar_left: " << scalar_left;
      }

      // Check that result type matches expected output type
      if (gpu_result.dtype() != expected_cpu.dtype()) {
        return ::testing::AssertionFailure()
               << "Mod result type mismatch: Expected " << expected_cpu.dtype() << " but got "
               << gpu_result.dtype();
      }

      return ::testing::AssertionSuccess();
    }

    // Get dtype list excluding complex numbers (Mod doesn't support complex)
    std::vector<unsigned int> GetModSupportedTypes() {
      std::vector<unsigned int> supported_types;
      for (auto dtype : test::dtype_list) {
        if (dtype != Type.ComplexDouble && dtype != Type.ComplexFloat) {
          supported_types.push_back(dtype);
        }
      }
      return supported_types;
    }

    std::vector<std::vector<cytnx_uint64>> GetTestShapes() {
      std::vector<std::vector<cytnx_uint64>> all_shapes;

      auto shapes_1d = test::GenerateTestShapes(1, 1, 1024, 4);
      auto shapes_2d = test::GenerateTestShapes(2, 1, 512, 4);
      auto shapes_3d = test::GenerateTestShapes(3, 1, 64, 4);
      auto shapes_4d = test::GenerateTestShapes(4, 1, 16, 4);

      all_shapes.insert(all_shapes.end(), shapes_1d.begin(), shapes_1d.end());
      all_shapes.insert(all_shapes.end(), shapes_2d.begin(), shapes_2d.end());
      all_shapes.insert(all_shapes.end(), shapes_3d.begin(), shapes_3d.end());
      all_shapes.insert(all_shapes.end(), shapes_4d.begin(), shapes_4d.end());

      return all_shapes;
    }

    class ModTestAllShapes : public ::testing::TestWithParam<std::vector<cytnx_uint64>> {};

    // Test tensor-to-tensor modulo
    TEST_P(ModTestAllShapes, GpuTensorModTensorAllTypes) {
      const std::vector<cytnx_uint64>& shape = GetParam();
      auto supported_types = GetModSupportedTypes();

      for (auto dtype : supported_types) {
        SCOPED_TRACE("Testing Mod(tensor, tensor) with shape: " + ::testing::PrintToString(shape) +
                     " and dtype: " + std::to_string(dtype));

        Tensor gpu_tensor1 = Tensor(shape, dtype, Device.cuda);
        Tensor gpu_tensor2 = Tensor(shape, dtype, Device.cuda);

        test::InitTensorUniform(gpu_tensor1);
        test::InitTensorUniform(gpu_tensor2);

        // Ensure divisor is not zero by adding a constant
        if (dtype == Type.Bool) {
          // For Bool, set divisor to 1 (true) to avoid division by zero
          gpu_tensor2 = ones(shape, dtype, Device.cuda);
        } else {
          // For numeric types, add a non-zero constant
          gpu_tensor2 = gpu_tensor2 + 3.0;
        }

        Tensor gpu_result = linalg::Mod(gpu_tensor1, gpu_tensor2);
        EXPECT_TRUE(CheckModResult(gpu_result, gpu_tensor1, gpu_tensor2));

        Tensor gpu_result_member = gpu_tensor1.Mod(gpu_tensor2);
        EXPECT_TRUE(CheckModResult(gpu_result_member, gpu_tensor1, gpu_tensor2));

        Tensor gpu_result_operator = (gpu_tensor1 % gpu_tensor2);
        EXPECT_TRUE(CheckModResult(gpu_result_operator, gpu_tensor1, gpu_tensor2));
      }
    }

    // Test scalar-to-tensor modulo
    TEST_P(ModTestAllShapes, GpuScalarModTensorAllTypes) {
      const std::vector<cytnx_uint64>& shape = GetParam();
      auto supported_types = GetModSupportedTypes();

      for (auto dtype : supported_types) {
        SCOPED_TRACE("Testing Mod(scalar, tensor) with shape: " + ::testing::PrintToString(shape) +
                     " and dtype: " + std::to_string(dtype));

        Tensor gpu_tensor = Tensor(shape, dtype, Device.cuda);
        test::InitTensorUniform(gpu_tensor);

        // Ensure divisor is not zero by adding a constant
        if (dtype == Type.Bool) {
          gpu_tensor = ones(shape, dtype, Device.cuda);
        } else {
          gpu_tensor = gpu_tensor + 1.2;
        }

        const cytnx_double scalar = 2.2;

        Tensor gpu_result = linalg::Mod(scalar, gpu_tensor);
        EXPECT_TRUE(CheckModScalarResult(gpu_result, gpu_tensor, scalar, true));

        Tensor gpu_result_operator = (scalar % gpu_tensor);
        EXPECT_TRUE(CheckModScalarResult(gpu_result_operator, gpu_tensor, scalar, true));
      }
    }

    // Test tensor-to-scalar modulo
    TEST_P(ModTestAllShapes, GpuTensorModScalarAllTypes) {
      const std::vector<cytnx_uint64>& shape = GetParam();
      auto supported_types = GetModSupportedTypes();

      for (auto dtype : supported_types) {
        SCOPED_TRACE("Testing Mod(tensor, scalar) with shape: " + ::testing::PrintToString(shape) +
                     " and dtype: " + std::to_string(dtype));

        Tensor gpu_tensor = Tensor(shape, dtype, Device.cuda);
        test::InitTensorUniform(gpu_tensor);

        // Use a non-zero scalar as divisor
        const cytnx_double scalar = 2.3;

        Tensor gpu_result = linalg::Mod(gpu_tensor, scalar);
        EXPECT_TRUE(CheckModScalarResult(gpu_result, gpu_tensor, scalar, false));

        Tensor gpu_result_member = gpu_tensor.Mod(scalar);
        EXPECT_TRUE(CheckModScalarResult(gpu_result_member, gpu_tensor, scalar, false));

        Tensor gpu_result_operator = (gpu_tensor % scalar);
        EXPECT_TRUE(CheckModScalarResult(gpu_result_operator, gpu_tensor, scalar, false));
      }
    }

    INSTANTIATE_TEST_SUITE_P(ModTests, ModTestAllShapes, ::testing::ValuesIn(GetTestShapes()));

  }  // namespace gpu_test
}  // namespace cytnx
