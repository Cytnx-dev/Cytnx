#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

// GPU Kron coverage. #1003 retired the Type_list_gpu / type_promote_gpu_t promotion
// trait: Kron's GPU path now computes the promoted type with the shared host
// type_promote_from_pointer_t and maps to the CUDA-native kernel type via to_cuda_t
// at the launch boundary. These tests move CPU-built inputs to the GPU, run Kron
// there, and check the result against independent hand-computed values -- especially
// the ComplexFloat x Double -> ComplexDouble promotion, which exercises exactly that
// path. Kron(a, b) for rank-1 a (len m) and b (len n) is a length-(m*n) vector with
// element (i*n + j) = a_i * b_j.

namespace cytnx {
  namespace gpu_test {
    namespace {

      TEST(Kron, GpuRealValues) {
        Tensor a = zeros({3}, Type.Double);
        a.at<cytnx_double>({0}) = 1;
        a.at<cytnx_double>({1}) = 2;
        a.at<cytnx_double>({2}) = 3;
        Tensor b = zeros({2}, Type.Double);
        b.at<cytnx_double>({0}) = 4;
        b.at<cytnx_double>({1}) = 5;

        Tensor out = linalg::Kron(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);
        ASSERT_EQ(out.shape(), std::vector<cytnx_uint64>({6}));
        ASSERT_EQ(out.dtype(), Type.Double);
        const double expect[6] = {4, 5, 8, 10, 12, 15};  // a_i * b_j, row-major (i, j)
        for (cytnx_uint64 k = 0; k < 6; ++k) EXPECT_DOUBLE_EQ(out.at<cytnx_double>({k}), expect[k]);
      }

      TEST(Kron, GpuComplexFloatDoublePromotesToComplexDouble) {
        // Exercises the retired-type_promote_gpu_t path: the promoted dtype differs
        // from both operands and the wider complex output must hold full precision.
        Tensor a = zeros({2}, Type.ComplexFloat);
        a.at<cytnx_complex64>({0}) = cytnx_complex64(1, 1);
        a.at<cytnx_complex64>({1}) = cytnx_complex64(2, 0);
        Tensor b = zeros({2}, Type.Double);
        b.at<cytnx_double>({0}) = 3;
        b.at<cytnx_double>({1}) = 0.5;

        Tensor out = linalg::Kron(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);
        ASSERT_EQ(out.dtype(), Type.ComplexDouble);
        ASSERT_EQ(out.shape(), std::vector<cytnx_uint64>({4}));
        // (1+1i)*3, (1+1i)*0.5, (2+0i)*3, (2+0i)*0.5
        const cytnx_complex128 expect[4] = {{3, 3}, {0.5, 0.5}, {6, 0}, {1, 0}};
        for (cytnx_uint64 k = 0; k < 4; ++k) {
          const cytnx_complex128 v = out.at<cytnx_complex128>({k});
          EXPECT_DOUBLE_EQ(v.real(), expect[k].real());
          EXPECT_DOUBLE_EQ(v.imag(), expect[k].imag());
        }
      }

      TEST(Kron, GpuInt16Values) {
        Tensor a = zeros({2}, Type.Int16);
        a.at<cytnx_int16>({0}) = 3;
        a.at<cytnx_int16>({1}) = -2;
        Tensor b = zeros({2}, Type.Int16);
        b.at<cytnx_int16>({0}) = 4;
        b.at<cytnx_int16>({1}) = 5;

        Tensor out = linalg::Kron(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);
        ASSERT_EQ(out.dtype(), Type.Int16);
        const cytnx_int16 expect[4] = {12, 15, -8, -10};
        for (cytnx_uint64 k = 0; k < 4; ++k) EXPECT_EQ(out.at<cytnx_int16>({k}), expect[k]);
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
