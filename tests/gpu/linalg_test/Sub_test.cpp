#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

namespace cytnx {
  namespace gpu_test {
    namespace {

      ::testing::AssertionResult CheckSubResult(const Tensor& gpu_result, const Tensor& left_tensor,
                                                const Tensor& right_tensor);

      ::testing::AssertionResult CheckSubScalarResult(const Tensor& gpu_result,
                                                      const Tensor& tensor,
                                                      const cytnx_double& scalar,
                                                      bool scalar_left = false);

      std::vector<std::vector<cytnx_uint64>> GetTestShapes();

      static cytnx_double GetTolerance(const unsigned int& dtype);

      class SubTestAllShapes : public ::testing::TestWithParam<std::vector<cytnx_uint64>> {};

      // Test tensor-to-tensor subtraction
      TEST_P(SubTestAllShapes, GpuTensorSubTensorAllTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto dtype : dtype_list) {
          if (dtype == Type.Bool) {
            continue;
          }

          SCOPED_TRACE("Testing Sub(tensor, tensor) with shape: " +
                       ::testing::PrintToString(shape) + " and dtype: " + std::to_string(dtype));

          Tensor gpu_tensor1 = Tensor(shape, dtype).to(Device.cuda);
          Tensor gpu_tensor2 = Tensor(shape, dtype).to(Device.cuda);
          InitTensorUniform(gpu_tensor1);
          InitTensorUniform(gpu_tensor2);

          Tensor gpu_result = linalg::Sub(gpu_tensor1, gpu_tensor2);
          EXPECT_TRUE(CheckSubResult(gpu_result, gpu_tensor1, gpu_tensor2));

          Tensor gpu_result_member = gpu_tensor1.Sub(gpu_tensor2);
          EXPECT_TRUE(CheckSubResult(gpu_result_member, gpu_tensor1, gpu_tensor2));

          Tensor gpu_result_operator = gpu_tensor1 - gpu_tensor2;
          EXPECT_TRUE(CheckSubResult(gpu_result_operator, gpu_tensor1, gpu_tensor2));
        }
      }

      // Test scalar-to-tensor subtraction
      TEST_P(SubTestAllShapes, GpuScalarSubTensorAllTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto dtype : dtype_list) {
          if (dtype == Type.Bool) {
            continue;
          }

          SCOPED_TRACE("Testing Sub(scalar, tensor) with shape: " +
                       ::testing::PrintToString(shape) + " and dtype: " + std::to_string(dtype));

          Tensor gpu_tensor = Tensor(shape, dtype).to(Device.cuda);
          InitTensorUniform(gpu_tensor);
          cytnx_double scalar = 2.3;

          Tensor gpu_result = linalg::Sub(scalar, gpu_tensor);
          EXPECT_TRUE(CheckSubScalarResult(gpu_result, gpu_tensor, scalar, true));

          Tensor gpu_result_operator = scalar - gpu_tensor;
          EXPECT_TRUE(CheckSubScalarResult(gpu_result_operator, gpu_tensor, scalar, true));
        }
      }

      // Test tensor-to-scalar subtraction
      TEST_P(SubTestAllShapes, GpuTensorSubScalarAllTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto dtype : dtype_list) {
          if (dtype == Type.Bool) {
            continue;
          }

          SCOPED_TRACE("Testing Sub(tensor, scalar) with shape: " +
                       ::testing::PrintToString(shape) + " and dtype: " + std::to_string(dtype));

          Tensor gpu_tensor = Tensor(shape, dtype).to(Device.cuda);
          InitTensorUniform(gpu_tensor);
          cytnx_double scalar = 2.3;

          Tensor gpu_result = linalg::Sub(gpu_tensor, scalar);
          EXPECT_TRUE(CheckSubScalarResult(gpu_result, gpu_tensor, scalar, false));

          Tensor gpu_result_member = gpu_tensor.Sub(scalar);
          EXPECT_TRUE(CheckSubScalarResult(gpu_result_member, gpu_tensor, scalar, false));

          Tensor gpu_result_operator = gpu_tensor - scalar;
          EXPECT_TRUE(CheckSubScalarResult(gpu_result_operator, gpu_tensor, scalar, false));
        }
      }

      // Test in-place tensor subtraction
      TEST_P(SubTestAllShapes, GpuTensorIsubAllTypes) {
        const std::vector<cytnx_uint64>& shape = GetParam();

        for (auto dtype : dtype_list) {
          if (dtype == Type.Bool) {
            continue;
          }

          SCOPED_TRACE("Testing iSub(tensor, tensor) with shape: " +
                       ::testing::PrintToString(shape) + " and dtype: " + std::to_string(dtype));

          Tensor gpu_tensor1 = Tensor(shape, dtype).to(Device.cuda);
          Tensor gpu_tensor2 = Tensor(shape, dtype).to(Device.cuda);
          InitTensorUniform(gpu_tensor1);
          InitTensorUniform(gpu_tensor2);

          Tensor original_gpu_tensor1 = gpu_tensor1.clone();
          Tensor original_gpu_tensor2 = gpu_tensor2.clone();

          linalg::iSub(gpu_tensor1, gpu_tensor2);
          EXPECT_TRUE(CheckSubResult(gpu_tensor1, original_gpu_tensor1, original_gpu_tensor2));

          Tensor gpu_tensor1_op = original_gpu_tensor1.clone();
          gpu_tensor1_op -= original_gpu_tensor2;
          EXPECT_TRUE(CheckSubResult(gpu_tensor1_op, original_gpu_tensor1, original_gpu_tensor2));
        }
      }

      TEST(SubMixedDtypeTest, GpuTensorSubTensorMixedUnsignedSignedTypePromote) {
        Tensor lhs = arange(0, 6, 1, Type.Uint32).reshape({2, 3});
        Tensor rhs = arange(0, 6, 1, Type.Int16).reshape({2, 3});
        lhs = lhs.to(Device.cuda);
        rhs = rhs.to(Device.cuda);

        Tensor gpu_result = linalg::Sub(lhs, rhs);
        Tensor expected_cpu = linalg::Sub(lhs.to(Device.cpu), rhs.to(Device.cpu));
        Tensor gpu_result_cpu = gpu_result.to(Device.cpu);

        EXPECT_EQ(gpu_result.dtype(), expected_cpu.dtype());
        EXPECT_TRUE(AreNearlyEqTensor(gpu_result_cpu, expected_cpu, 1e-6));
      }

      TEST(SubMixedDtypeTest, GpuScalarSubTensorMixedUnsignedSignedTypePromote) {
        const cytnx_uint32 scalar = 5;
        Tensor rhs = arange(0, 6, 1, Type.Int16, Device.cuda).reshape({2, 3});

        Tensor gpu_result_l = linalg::Sub(scalar, rhs);
        Tensor gpu_result_r = linalg::Sub(rhs, scalar);

        Tensor rhs_cpu = rhs.to(Device.cpu);
        Tensor expected_l = linalg::Sub(scalar, rhs_cpu);
        Tensor expected_r = linalg::Sub(rhs_cpu, scalar);
        Tensor gpu_result_l_cpu = gpu_result_l.to(Device.cpu);
        Tensor gpu_result_r_cpu = gpu_result_r.to(Device.cpu);

        EXPECT_EQ(gpu_result_l.dtype(), expected_l.dtype());
        EXPECT_EQ(gpu_result_r.dtype(), expected_r.dtype());
        EXPECT_TRUE(AreNearlyEqTensor(gpu_result_l_cpu, expected_l, 1e-6));
        EXPECT_TRUE(AreNearlyEqTensor(gpu_result_r_cpu, expected_r, 1e-6));
      }

      // The #984/#999 promotion discriminator on the GPU Sub path (#1003): ComplexFloat
      // (-) Double must produce ComplexDouble -- the higher-precision complex -- with the
      // subtraction computed and stored through that output type. The sweeps above pair a
      // dtype with itself and the mixed tests above cover only signed/unsigned, so this
      // row was untested here.
      //
      // The Double operands are deliberately values float32 cannot represent (0.1, 3.3).
      // An implementation that reports ComplexDouble but computes through ComplexFloat
      // rounds them first and lands ~1e-8 away -- four orders of magnitude past the
      // tolerance below, so it fails loudly. The ComplexFloat components stay float-exact
      // (1.5, -2.5, 0, 4) so float -> double widening of that operand is exact and the
      // only precision question under test is the one above.
      //
      // Both operand orders are checked because subtraction does not commute: a path that
      // silently swapped or narrowed one side would survive a single-order test. Expected
      // values are evaluated here in host double arithmetic rather than taken from the CPU
      // Sub path, so a shared promotion bug cannot hide.
      TEST(SubMixedDtypeTest, GpuComplexFloatDoublePromotesToComplexDouble) {
        Tensor cf = zeros({2}, Type.ComplexFloat);
        cf.at<cytnx_complex64>({0}) = cytnx_complex64(1.5f, -2.5f);
        cf.at<cytnx_complex64>({1}) = cytnx_complex64(0.0f, 4.0f);
        Tensor d = zeros({2}, Type.Double);
        d.at<cytnx_double>({0}) = 0.1;
        d.at<cytnx_double>({1}) = 3.3;

        {
          SCOPED_TRACE("ComplexFloat - Double");
          Tensor out = linalg::Sub(cf.to(Device.cuda), d.to(Device.cuda)).to(Device.cpu);
          ASSERT_EQ(out.dtype(), Type.ComplexDouble);
          Tensor expected = zeros({2}, Type.ComplexDouble);
          expected.at<cytnx_complex128>({0}) = cytnx_complex128(1.5 - 0.1, -2.5);
          expected.at<cytnx_complex128>({1}) = cytnx_complex128(0.0 - 3.3, 4.0);
          EXPECT_TRUE(AreNearlyEqTensor(out, expected, 1e-12));
        }
        {
          SCOPED_TRACE("Double - ComplexFloat");
          Tensor out = linalg::Sub(d.to(Device.cuda), cf.to(Device.cuda)).to(Device.cpu);
          ASSERT_EQ(out.dtype(), Type.ComplexDouble);
          Tensor expected = zeros({2}, Type.ComplexDouble);
          expected.at<cytnx_complex128>({0}) = cytnx_complex128(0.1 - 1.5, 2.5);
          expected.at<cytnx_complex128>({1}) = cytnx_complex128(3.3 - 0.0, -4.0);
          EXPECT_TRUE(AreNearlyEqTensor(out, expected, 1e-12));
        }
      }

      INSTANTIATE_TEST_SUITE_P(SubTests, SubTestAllShapes, ::testing::ValuesIn(GetTestShapes()));

      ::testing::AssertionResult CheckSubResult(const Tensor& gpu_result, const Tensor& left_tensor,
                                                const Tensor& right_tensor) {
        // Compare CUDA Sub result against CPU Sub result
        Tensor left_cpu = left_tensor.to(Device.cpu);
        Tensor right_cpu = right_tensor.to(Device.cpu);
        Tensor expected_cpu = linalg::Sub(left_cpu, right_cpu);
        Tensor gpu_result_cpu = gpu_result.to(Device.cpu);

        cytnx_double tolerance = GetTolerance(gpu_result.dtype());

        if (!AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
          return ::testing::AssertionFailure()
                 << "Sub result mismatch: CUDA Sub result differs from CPU Sub result. "
                 << "Left dtype: " << left_tensor.dtype()
                 << ", Right dtype: " << right_tensor.dtype() << ", tolerance used: " << tolerance;
        }

        return ::testing::AssertionSuccess();
      }

      ::testing::AssertionResult CheckSubScalarResult(const Tensor& gpu_result,
                                                      const Tensor& tensor,
                                                      const cytnx_double& scalar,
                                                      bool scalar_left) {
        // Compare CUDA Sub result against CPU Sub result
        Tensor tensor_cpu = tensor.to(Device.cpu);
        Tensor expected_cpu;

        if (scalar_left) {
          expected_cpu = linalg::Sub(scalar, tensor_cpu);
        } else {
          expected_cpu = linalg::Sub(tensor_cpu, scalar);
        }

        Tensor gpu_result_cpu = gpu_result.to(Device.cpu);

        cytnx_double tolerance = GetTolerance(gpu_result.dtype());

        if (!AreNearlyEqTensor(gpu_result_cpu, expected_cpu, tolerance)) {
          return ::testing::AssertionFailure()
                 << "Sub scalar result mismatch: CUDA Sub result differs from CPU Sub result. "
                 << "Tensor dtype: " << tensor.dtype() << ", scalar: " << scalar
                 << ", scalar_left: " << scalar_left << ", tolerance used: " << tolerance;
        }

        return ::testing::AssertionSuccess();
      }

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

      static cytnx_double GetTolerance(const unsigned int& dtype) {
        cytnx_double tolerance = 1e-6;
        return tolerance;
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
