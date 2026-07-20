#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

// Coverage for the #1003 step-4 change that passes the non-contiguous layout
// (output strides, per-operand strides and inverse mappers) to the GPU elementwise
// kernels BY VALUE instead of cuMalloc_gpu'ing/cudaMemcpy'ing/cudaFree'ing five
// device arrays on every call (shared cuNonContigLayout.cuh).
//
// Reachability of the out-of-place non-contiguous kernel: only Add and Sub feed a
// non-contiguous tensor(op)tensor to the GPU dispatch; Mul/Div/Cpr still reject a
// non-contiguous operand at the front end (contiguous-ize first), so their
// non-contiguous branches are unreachable. The out-of-place kernel these tests drive
// (via Add/Sub) is the same shared tn_kernel_nonconti the whole cuArithmeticDispatch
// family instantiates, and it shares MakeGpuNonContigLayout / ComputeGpuNonContigIndices
// with the in-place kernel (already covered by InplacePromote_test). Inputs are built
// with CPU arange (then moved to the GPU) so the test data never depends on GPU kernels.
namespace cytnx {
  namespace gpu_test {
    namespace {

      // Out-of-place ops that accept a non-contiguous operand on the GPU: Add, Sub.
      Tensor ApplyBinary(int op, const Tensor& l, const Tensor& r) {
        return op == 0 ? linalg::Add(l, r) : linalg::Sub(l, r);
      }
      const char* OpName(int op) { return op == 0 ? "Add" : "Sub"; }

      // Out-of-place tensor(op)tensor with BOTH operands non-contiguous (permuted):
      // the logical elements sit at different physical offsets in each buffer, so the
      // kernel must gather lhs[Lidx] and rhs[Ridx] through the two inverse mappers. The
      // result is written contiguously at the logical index. Compare the GPU result
      // (dtype AND value) against the independent CPU path.
      TEST(NonContigElementwiseTest, GpuOutOfPlaceNoncontiguousMatchesCpu) {
        for (int op : {0, 2}) {
          for (auto dtype : dtype_list) {
            if (dtype == Type.Bool) continue;
            SCOPED_TRACE(std::string(OpName(op)) + " dtype=" + std::to_string(dtype));

            Tensor gpu_l = arange(1, 7, 1, dtype).reshape({3, 2}).permute({1, 0}).to(Device.cuda);
            Tensor gpu_r = arange(2, 8, 1, dtype).reshape({3, 2}).permute({1, 0}).to(Device.cuda);
            ASSERT_FALSE(gpu_l.is_contiguous());
            ASSERT_FALSE(gpu_r.is_contiguous());
            Tensor cpu_l = gpu_l.to(Device.cpu);
            Tensor cpu_r = gpu_r.to(Device.cpu);

            Tensor gpu_out = ApplyBinary(op, gpu_l, gpu_r);
            Tensor cpu_out = ApplyBinary(op, cpu_l, cpu_r);

            EXPECT_EQ(gpu_out.dtype(), cpu_out.dtype());
            EXPECT_TRUE(AreNearlyEqTensor(gpu_out.to(Device.cpu), cpu_out, 1e-5));
          }
        }
      }

      // Independent literal expected values (NOT the CPU oracle) for the out-of-place
      // non-contiguous mapper math, per CLAUDE.md: a bug in the shared gather would
      // otherwise be able to hide behind a GPU-vs-CPU comparison. Both operands are
      // permuted, so both Lidx and Ridx are non-trivial.
      TEST(NonContigElementwiseTest, GpuOutOfPlaceNoncontiguousLiteral) {
        // l logical = [[1,3,5],[2,4,6]], r logical = [[10,30,50],[20,40,60]]
        // (arange 3x2 permuted).
        Tensor l = arange(1, 7, 1, Type.Int64).reshape({3, 2}).permute({1, 0}).to(Device.cuda);
        Tensor r = arange(10, 70, 10, Type.Int64).reshape({3, 2}).permute({1, 0}).to(Device.cuda);
        ASSERT_FALSE(l.is_contiguous());
        ASSERT_FALSE(r.is_contiguous());

        // Add -> [[11,33,55],[22,44,66]]
        {
          Tensor got = linalg::Add(l, r).to(Device.cpu);
          EXPECT_EQ(got.dtype(), Type.Int64);
          const cytnx_int64 expect[2][3] = {{11, 33, 55}, {22, 44, 66}};
          for (cytnx_uint64 i = 0; i < 2; i++)
            for (cytnx_uint64 j = 0; j < 3; j++)
              EXPECT_EQ(got.at<cytnx_int64>({i, j}), expect[i][j]);
        }
        // Sub -> [[-9,-27,-45],[-18,-36,-54]]
        {
          Tensor got = linalg::Sub(l, r).to(Device.cpu);
          EXPECT_EQ(got.dtype(), Type.Int64);
          const cytnx_int64 expect[2][3] = {{-9, -27, -45}, {-18, -36, -54}};
          for (cytnx_uint64 i = 0; i < 2; i++)
            for (cytnx_uint64 j = 0; j < 3; j++)
              EXPECT_EQ(got.at<cytnx_int64>({i, j}), expect[i][j]);
        }

        // Mixed dtype: Int32 (permuted) + Double (permuted) -> Double, same values as the
        // Add case, pinning both the type promotion and the gather at once.
        {
          Tensor li = arange(1, 7, 1, Type.Int32).reshape({3, 2}).permute({1, 0}).to(Device.cuda);
          Tensor rd =
            arange(10, 70, 10, Type.Double).reshape({3, 2}).permute({1, 0}).to(Device.cuda);
          Tensor got = linalg::Add(li, rd).to(Device.cpu);
          EXPECT_EQ(got.dtype(), Type.Double);
          const cytnx_double expect[2][3] = {{11, 33, 55}, {22, 44, 66}};
          for (cytnx_uint64 i = 0; i < 2; i++)
            for (cytnx_uint64 j = 0; j < 3; j++)
              EXPECT_DOUBLE_EQ(got.at<cytnx_double>({i, j}), expect[i][j]);
        }
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
