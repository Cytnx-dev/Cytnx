#include "gtest/gtest.h"

#include "gpu_test_tools.h"
#include "cytnx.hpp"

namespace AddTest {

  ::testing::AssertionResult CheckAddResult(const cytnx::Tensor& gpu_result,
                                            const cytnx::Tensor& left_tensor,
                                            const cytnx::Tensor& right_tensor) {
    // Compare CUDA Add result against CPU Add result
    cytnx::Tensor left_cpu = left_tensor.to(cytnx::Device.cpu);
    cytnx::Tensor right_cpu = right_tensor.to(cytnx::Device.cpu);
    cytnx::Tensor expected_cpu = cytnx::linalg::Add(left_cpu, right_cpu);
    cytnx::Tensor gpu_result_cpu = gpu_result.to(cytnx::Device.cpu);

    cytnx::cytnx_double tolerance = 1e-6;

    if (!cytnx::TestTools::AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
      return ::testing::AssertionFailure()
             << "Add result mismatch: CUDA Add result differs from CPU Add result. "
             << "Left dtype: " << left_tensor.dtype() << ", Right dtype: " << right_tensor.dtype()
             << ", tolerance used: " << tolerance;
    }

    return ::testing::AssertionSuccess();
  }

  ::testing::AssertionResult CheckiAddResult(const cytnx::Tensor& gpu_result,
                                             const cytnx::Tensor& original_left,
                                             const cytnx::Tensor& original_right) {
    // Compare CUDA iAdd result against CPU iAdd result
    cytnx::Tensor left_cpu = original_left.to(cytnx::Device.cpu);
    cytnx::Tensor right_cpu = original_right.to(cytnx::Device.cpu);

    // Use iAdd on CPU to get expected result
    cytnx::linalg::iAdd(left_cpu, right_cpu);
    cytnx::Tensor expected_cpu = left_cpu;

    cytnx::Tensor gpu_result_cpu = gpu_result.to(cytnx::Device.cpu);

    cytnx::cytnx_double tolerance = 1e-6;

    if (!cytnx::TestTools::AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
      return ::testing::AssertionFailure()
             << "iAdd result mismatch: CUDA iAdd result differs from CPU iAdd result. "
             << "Left dtype: " << original_left.dtype()
             << ", Right dtype: " << original_right.dtype() << ", tolerance used: " << tolerance;
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

  // Helper to test scalar+tensor with specific scalar type
  template <typename ScalarType>
  ::testing::AssertionResult TestScalarAddTensor(const cytnx::Tensor& gpu_tensor,
                                                 ScalarType scalar_val, bool scalar_left) {
    cytnx::Tensor tensor_cpu = gpu_tensor.to(cytnx::Device.cpu);
    cytnx::Tensor gpu_result, expected_cpu;

    if (scalar_left) {
      gpu_result = cytnx::linalg::Add(scalar_val, gpu_tensor);
      expected_cpu = cytnx::linalg::Add(scalar_val, tensor_cpu);
    } else {
      gpu_result = cytnx::linalg::Add(gpu_tensor, scalar_val);
      expected_cpu = cytnx::linalg::Add(tensor_cpu, scalar_val);
    }

    cytnx::Tensor gpu_result_cpu = gpu_result.to(cytnx::Device.cpu);
    cytnx::cytnx_double tolerance = 1e-6;

    if (!cytnx::TestTools::AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
      return ::testing::AssertionFailure()
             << "Add scalar result mismatch: CUDA differs from CPU. "
             << "Tensor dtype: " << gpu_tensor.dtype() << ", scalar_left: " << scalar_left;
    }
    return ::testing::AssertionSuccess();
  }

  // Dispatch scalar test based on scalar dtype
  inline ::testing::AssertionResult DispatchScalarAddTest(const cytnx::Tensor& gpu_tensor,
                                                          unsigned int sdtype, bool scalar_left) {
    switch (sdtype) {
      case cytnx::Type.ComplexDouble:
        return TestScalarAddTensor(gpu_tensor, cytnx::cytnx_complex128(2.3, 1.1), scalar_left);
      case cytnx::Type.ComplexFloat:
        return TestScalarAddTensor(gpu_tensor, cytnx::cytnx_complex64(2.3f, 1.1f), scalar_left);
      case cytnx::Type.Double:
        return TestScalarAddTensor(gpu_tensor, cytnx::cytnx_double(2.3), scalar_left);
      case cytnx::Type.Float:
        return TestScalarAddTensor(gpu_tensor, cytnx::cytnx_float(2.3f), scalar_left);
      case cytnx::Type.Int64:
        return TestScalarAddTensor(gpu_tensor, cytnx::cytnx_int64(2), scalar_left);
      case cytnx::Type.Uint64:
        return TestScalarAddTensor(gpu_tensor, cytnx::cytnx_uint64(2), scalar_left);
      case cytnx::Type.Int32:
        return TestScalarAddTensor(gpu_tensor, cytnx::cytnx_int32(2), scalar_left);
      case cytnx::Type.Uint32:
        return TestScalarAddTensor(gpu_tensor, cytnx::cytnx_uint32(2), scalar_left);
      case cytnx::Type.Int16:
        return TestScalarAddTensor(gpu_tensor, cytnx::cytnx_int16(2), scalar_left);
      case cytnx::Type.Uint16:
        return TestScalarAddTensor(gpu_tensor, cytnx::cytnx_uint16(2), scalar_left);
      default:
        return ::testing::AssertionFailure() << "Unsupported scalar dtype: " << sdtype;
    }
  }

  class AddTestAllShapes : public ::testing::TestWithParam<std::vector<cytnx::cytnx_uint64>> {};

  // Test tensor-to-tensor addition with mixed types
  TEST_P(AddTestAllShapes, gpu_tensor_add_tensor_mixed_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto ldtype : cytnx::TestTools::dtype_list) {
      if (ldtype == cytnx::Type.Bool) continue;

      for (auto rdtype : cytnx::TestTools::dtype_list) {
        if (rdtype == cytnx::Type.Bool) continue;

        SCOPED_TRACE("Testing Add mixed types with shape: " + ::testing::PrintToString(shape) +
                     ", ldtype: " + std::to_string(ldtype) + ", rdtype: " + std::to_string(rdtype));

        cytnx::Tensor gpu_tensor1 = cytnx::Tensor(shape, ldtype).to(cytnx::Device.cuda);
        cytnx::Tensor gpu_tensor2 = cytnx::Tensor(shape, rdtype).to(cytnx::Device.cuda);
        cytnx::TestTools::InitTensorUniform(gpu_tensor1);
        cytnx::TestTools::InitTensorUniform(gpu_tensor2);

        cytnx::Tensor gpu_result = cytnx::linalg::Add(gpu_tensor1, gpu_tensor2);
        EXPECT_TRUE(CheckAddResult(gpu_result, gpu_tensor1, gpu_tensor2));

        cytnx::Tensor gpu_result_member = gpu_tensor1.Add(gpu_tensor2);
        EXPECT_TRUE(CheckAddResult(gpu_result_member, gpu_tensor1, gpu_tensor2));

        cytnx::Tensor gpu_result_op = gpu_tensor1 + gpu_tensor2;
        EXPECT_TRUE(CheckAddResult(gpu_result_op, gpu_tensor1, gpu_tensor2));
      }
    }
  }

  // Test scalar-to-tensor addition with mixed types
  TEST_P(AddTestAllShapes, gpu_scalar_add_tensor_mixed_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto sdtype : cytnx::TestTools::dtype_list) {
      if (sdtype == cytnx::Type.Bool) continue;

      for (auto tdtype : cytnx::TestTools::dtype_list) {
        if (tdtype == cytnx::Type.Bool) continue;

        SCOPED_TRACE("Testing Add(scalar, tensor) with shape: " + ::testing::PrintToString(shape) +
                     ", sdtype: " + std::to_string(sdtype) + ", tdtype: " + std::to_string(tdtype));

        cytnx::Tensor gpu_tensor = cytnx::Tensor(shape, tdtype).to(cytnx::Device.cuda);
        cytnx::TestTools::InitTensorUniform(gpu_tensor);

        EXPECT_TRUE(DispatchScalarAddTest(gpu_tensor, sdtype, true));
      }
    }
  }

  // Test tensor-to-scalar addition with mixed types
  TEST_P(AddTestAllShapes, gpu_tensor_add_scalar_mixed_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto tdtype : cytnx::TestTools::dtype_list) {
      if (tdtype == cytnx::Type.Bool) continue;

      for (auto sdtype : cytnx::TestTools::dtype_list) {
        if (sdtype == cytnx::Type.Bool) continue;

        SCOPED_TRACE("Testing Add(tensor, scalar) with shape: " + ::testing::PrintToString(shape) +
                     ", tdtype: " + std::to_string(tdtype) + ", sdtype: " + std::to_string(sdtype));

        cytnx::Tensor gpu_tensor = cytnx::Tensor(shape, tdtype).to(cytnx::Device.cuda);
        cytnx::TestTools::InitTensorUniform(gpu_tensor);

        EXPECT_TRUE(DispatchScalarAddTest(gpu_tensor, sdtype, false));
      }
    }
  }

  // Test in-place tensor addition with mixed types
  TEST_P(AddTestAllShapes, gpu_tensor_iadd_mixed_types) {
    const std::vector<cytnx::cytnx_uint64>& shape = GetParam();

    for (auto ldtype : cytnx::TestTools::dtype_list) {
      if (ldtype == cytnx::Type.Bool) continue;

      for (auto rdtype : cytnx::TestTools::dtype_list) {
        if (rdtype == cytnx::Type.Bool) continue;

        // L += R
        // Skip if R has higher precision than L (result can't be stored in L's type)
        unsigned int promoted_type = cytnx::Type.type_promote(ldtype, rdtype);
        if (promoted_type != ldtype) continue;

        SCOPED_TRACE("Testing iAdd mixed types with shape: " + ::testing::PrintToString(shape) +
                     ", ldtype: " + std::to_string(ldtype) + ", rdtype: " + std::to_string(rdtype));

        cytnx::Tensor gpu_tensor1 = cytnx::Tensor(shape, ldtype).to(cytnx::Device.cuda);
        cytnx::Tensor gpu_tensor2 = cytnx::Tensor(shape, rdtype).to(cytnx::Device.cuda);
        cytnx::TestTools::InitTensorUniform(gpu_tensor1);
        cytnx::TestTools::InitTensorUniform(gpu_tensor2);

        cytnx::Tensor original_gpu_tensor1 = gpu_tensor1.clone();
        cytnx::Tensor original_gpu_tensor2 = gpu_tensor2.clone();

        cytnx::linalg::iAdd(gpu_tensor1, gpu_tensor2);
        EXPECT_TRUE(CheckiAddResult(gpu_tensor1, original_gpu_tensor1, original_gpu_tensor2));

        cytnx::Tensor gpu_tensor1_op = original_gpu_tensor1.clone();
        gpu_tensor1_op += original_gpu_tensor2;
        EXPECT_TRUE(CheckiAddResult(gpu_tensor1_op, original_gpu_tensor1, original_gpu_tensor2));
      }
    }
  }

  INSTANTIATE_TEST_SUITE_P(AddTests, AddTestAllShapes, ::testing::ValuesIn(GetTestShapes()));

}  // namespace AddTest
