#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"
namespace cytnx {
  namespace {
    namespace AddTest {

      ::testing::AssertionResult CheckAddResult(const Tensor& gpu_result, const Tensor& left_tensor,
                                                const Tensor& right_tensor) {
        // Compare CUDA Add result against CPU Add result
        Tensor left_cpu = left_tensor.to(Device.cpu);
        Tensor right_cpu = right_tensor.to(Device.cpu);
        Tensor expected_cpu = linalg::Add(left_cpu, right_cpu);
        Tensor gpu_result_cpu = gpu_result.to(Device.cpu);

        cytnx_double tolerance = 1e-6;

        if (!test::AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
          return ::testing::AssertionFailure()
                 << "Add result mismatch: CUDA Add result differs from CPU Add result. "
                 << "Left dtype: " << left_tensor.dtype()
                 << ", Right dtype: " << right_tensor.dtype() << ", tolerance used: " << tolerance;
        }

        return ::testing::AssertionSuccess();
      }

      ::testing::AssertionResult CheckiAddResult(const Tensor& gpu_result,
                                                 const Tensor& original_left,
                                                 const Tensor& original_right) {
        // Compare CUDA iAdd result against CPU iAdd result
        Tensor left_cpu = original_left.to(Device.cpu);
        Tensor right_cpu = original_right.to(Device.cpu);

        // Use iAdd on CPU to get expected result
        linalg::iAdd(left_cpu, right_cpu);
        Tensor expected_cpu = left_cpu;

        Tensor gpu_result_cpu = gpu_result.to(Device.cpu);

        cytnx_double tolerance = 1e-6;

        if (!test::AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
          return ::testing::AssertionFailure()
                 << "iAdd result mismatch: CUDA iAdd result differs from CPU iAdd result. "
                 << "Left dtype: " << original_left.dtype()
                 << ", Right dtype: " << original_right.dtype()
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

      // Helper to test scalar+tensor with specific scalar type
      template <typename ScalarType>
      ::testing::AssertionResult TestScalarAddTensor(const Tensor& gpu_tensor,
                                                     ScalarType scalar_val, bool scalar_left) {
        Tensor tensor_cpu = gpu_tensor.to(Device.cpu);
        Tensor gpu_result, expected_cpu;

        if (scalar_left) {
          gpu_result = linalg::Add(scalar_val, gpu_tensor);
          expected_cpu = linalg::Add(scalar_val, tensor_cpu);
        } else {
          gpu_result = linalg::Add(gpu_tensor, scalar_val);
          expected_cpu = linalg::Add(tensor_cpu, scalar_val);
        }

        Tensor gpu_result_cpu = gpu_result.to(Device.cpu);
        cytnx_double tolerance = 1e-6;

        if (!test::AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
          return ::testing::AssertionFailure()
                 << "Add scalar result mismatch: CUDA differs from CPU. "
                 << "Tensor dtype: " << gpu_tensor.dtype() << ", scalar_left: " << scalar_left;
        }
        return ::testing::AssertionSuccess();
      }

      // Dispatch scalar test based on scalar dtype
      inline ::testing::AssertionResult DispatchScalarAddTest(const Tensor& gpu_tensor,
                                                              unsigned int sdtype,
                                                              bool scalar_left) {
        switch (sdtype) {
          case Type.ComplexDouble:
            return TestScalarAddTensor(gpu_tensor, cytnx_complex128(2.3, 1.1), scalar_left);
          case Type.ComplexFloat:
            return TestScalarAddTensor(gpu_tensor, cytnx_complex64(2.3f, 1.1f), scalar_left);
          case Type.Double:
            return TestScalarAddTensor(gpu_tensor, cytnx_double(2.3), scalar_left);
          case Type.Float:
            return TestScalarAddTensor(gpu_tensor, cytnx_float(2.3f), scalar_left);
          case Type.Int64:
            return TestScalarAddTensor(gpu_tensor, cytnx_int64(2), scalar_left);
          case Type.Uint64:
            return TestScalarAddTensor(gpu_tensor, cytnx_uint64(2), scalar_left);
          case Type.Int32:
            return TestScalarAddTensor(gpu_tensor, cytnx_int32(2), scalar_left);
          case Type.Uint32:
            return TestScalarAddTensor(gpu_tensor, cytnx_uint32(2), scalar_left);
          case Type.Int16:
            return TestScalarAddTensor(gpu_tensor, cytnx_int16(2), scalar_left);
          case Type.Uint16:
            return TestScalarAddTensor(gpu_tensor, cytnx_uint16(2), scalar_left);
          default:
            return ::testing::AssertionFailure() << "Unsupported scalar dtype: " << sdtype;
        }
      }

      class AddTestAllShapes : public ::testing::TestWithParam<std::vector<cytnx_uint64>> {};

      // Test tensor-to-tensor addition with mixed types
      TEST_P(AddTestAllShapes, GpuTensorAddTensorMixedTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto ldtype : test::dtype_list) {
          if (ldtype == Type.Bool) continue;

          for (auto rdtype : test::dtype_list) {
            if (rdtype == Type.Bool) continue;

            SCOPED_TRACE("Testing Add mixed types with shape: " + ::testing::PrintToString(shape) +
                         ", ldtype: " + std::to_string(ldtype) +
                         ", rdtype: " + std::to_string(rdtype));

            Tensor gpu_tensor1 = Tensor(shape, ldtype).to(Device.cuda);
            Tensor gpu_tensor2 = Tensor(shape, rdtype).to(Device.cuda);
            test::InitTensorUniform(gpu_tensor1);
            test::InitTensorUniform(gpu_tensor2);

            Tensor gpu_result = linalg::Add(gpu_tensor1, gpu_tensor2);
            EXPECT_TRUE(CheckAddResult(gpu_result, gpu_tensor1, gpu_tensor2));

            Tensor gpu_result_member = gpu_tensor1.Add(gpu_tensor2);
            EXPECT_TRUE(CheckAddResult(gpu_result_member, gpu_tensor1, gpu_tensor2));

            Tensor gpu_result_op = gpu_tensor1 + gpu_tensor2;
            EXPECT_TRUE(CheckAddResult(gpu_result_op, gpu_tensor1, gpu_tensor2));
          }
        }
      }

      // Test scalar-to-tensor addition with mixed types
      TEST_P(AddTestAllShapes, GpuScalarAddTensorMixedTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto sdtype : test::dtype_list) {
          if (sdtype == Type.Bool) continue;

          for (auto tdtype : test::dtype_list) {
            if (tdtype == Type.Bool) continue;

            SCOPED_TRACE(
              "Testing Add(scalar, tensor) with shape: " + ::testing::PrintToString(shape) +
              ", sdtype: " + std::to_string(sdtype) + ", tdtype: " + std::to_string(tdtype));

            Tensor gpu_tensor = Tensor(shape, tdtype).to(Device.cuda);
            test::InitTensorUniform(gpu_tensor);

            EXPECT_TRUE(DispatchScalarAddTest(gpu_tensor, sdtype, true));
          }
        }
      }

      // Test tensor-to-scalar addition with mixed types
      TEST_P(AddTestAllShapes, GpuTensorAddScalarMixedTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto tdtype : test::dtype_list) {
          if (tdtype == Type.Bool) continue;

          for (auto sdtype : test::dtype_list) {
            if (sdtype == Type.Bool) continue;

            SCOPED_TRACE(
              "Testing Add(tensor, scalar) with shape: " + ::testing::PrintToString(shape) +
              ", tdtype: " + std::to_string(tdtype) + ", sdtype: " + std::to_string(sdtype));

            Tensor gpu_tensor = Tensor(shape, tdtype).to(Device.cuda);
            test::InitTensorUniform(gpu_tensor);

            EXPECT_TRUE(DispatchScalarAddTest(gpu_tensor, sdtype, false));
          }
        }
      }

      // Test in-place tensor addition with mixed types
      TEST_P(AddTestAllShapes, GpuTensorIaddMixedTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto ldtype : test::dtype_list) {
          if (ldtype == Type.Bool) continue;

          for (auto rdtype : test::dtype_list) {
            if (rdtype == Type.Bool) continue;

            // L += R
            // Skip if R has higher precision than L (result can't be stored in L's type)
            unsigned int promoted_type = Type.type_promote(ldtype, rdtype);
            if (promoted_type != ldtype) continue;

            SCOPED_TRACE("Testing iAdd mixed types with shape: " + ::testing::PrintToString(shape) +
                         ", ldtype: " + std::to_string(ldtype) +
                         ", rdtype: " + std::to_string(rdtype));

            Tensor gpu_tensor1 = Tensor(shape, ldtype).to(Device.cuda);
            Tensor gpu_tensor2 = Tensor(shape, rdtype).to(Device.cuda);
            test::InitTensorUniform(gpu_tensor1);
            test::InitTensorUniform(gpu_tensor2);

            Tensor original_gpu_tensor1 = gpu_tensor1.clone();
            Tensor original_gpu_tensor2 = gpu_tensor2.clone();

            linalg::iAdd(gpu_tensor1, gpu_tensor2);
            EXPECT_TRUE(CheckiAddResult(gpu_tensor1, original_gpu_tensor1, original_gpu_tensor2));

            Tensor gpu_tensor1_op = original_gpu_tensor1.clone();
            gpu_tensor1_op += original_gpu_tensor2;
            EXPECT_TRUE(
              CheckiAddResult(gpu_tensor1_op, original_gpu_tensor1, original_gpu_tensor2));
          }
        }
      }

      TEST(AddMixedDtypeTest, GpuTensorAddTensorMixedUnsignedSignedTypePromote) {
        Tensor lhs = arange(0, 6, 1, Type.Uint32).reshape({2, 3});
        Tensor rhs = arange(0, 6, 1, Type.Int16).reshape({2, 3});
        lhs = lhs.to(Device.cuda);
        rhs = rhs.to(Device.cuda);

        Tensor gpu_result = linalg::Add(lhs, rhs);
        Tensor expected_cpu = linalg::Add(lhs.to(Device.cpu), rhs.to(Device.cpu));
        Tensor gpu_result_cpu = gpu_result.to(Device.cpu);

        EXPECT_EQ(gpu_result.dtype(), expected_cpu.dtype());
        EXPECT_TRUE(test::AreNearlyEqTensor(gpu_result_cpu, expected_cpu, 1e-6));
      }

      TEST(AddMixedDtypeTest, GpuScalarAddTensorMixedUnsignedSignedTypePromote) {
        const cytnx_uint32 scalar = 5;
        Tensor rhs = arange(0, 6, 1, Type.Int16, Device.cuda).reshape({2, 3});

        Tensor gpu_result_l = linalg::Add(scalar, rhs);
        Tensor gpu_result_r = linalg::Add(rhs, scalar);

        Tensor rhs_cpu = rhs.to(Device.cpu);
        Tensor expected_l = linalg::Add(scalar, rhs_cpu);
        Tensor expected_r = linalg::Add(rhs_cpu, scalar);
        Tensor gpu_result_l_cpu = gpu_result_l.to(Device.cpu);
        Tensor gpu_result_r_cpu = gpu_result_r.to(Device.cpu);

        EXPECT_EQ(gpu_result_l.dtype(), expected_l.dtype());
        EXPECT_EQ(gpu_result_r.dtype(), expected_r.dtype());
        EXPECT_TRUE(test::AreNearlyEqTensor(gpu_result_l_cpu, expected_l, 1e-6));
        EXPECT_TRUE(test::AreNearlyEqTensor(gpu_result_r_cpu, expected_r, 1e-6));
      }

      INSTANTIATE_TEST_SUITE_P(AddTests, AddTestAllShapes, ::testing::ValuesIn(GetTestShapes()));

    }  // namespace AddTest

  }  // namespace
}  // namespace cytnx
