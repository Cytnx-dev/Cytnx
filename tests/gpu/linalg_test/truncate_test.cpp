#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

namespace cytnx {
  namespace gpu_test {
    namespace {

      // GPU regression tests for the truncated SVD family (Svd_truncate / Gesvd_truncate /
      // Rsvd). These mirror the CPU flag-combination tests and exercise the refactored
      // cudaMemcpyTruncation: the packed Svd/Gesvd output is [S, U?, vT?] and indexing it
      // positionally used to read out-of-range / the wrong matrix when is_U / is_vT / is_UvT were
      // false.

      // Singular values are gauge-invariant; compare them on the host with a relative tolerance
      // (LAPACK/cuSOLVER can differ slightly across job modes). U/vT are sign/gauge dependent, so
      // the tests only assert their truncated shapes.
      static bool SingularValsClose(const Tensor& a, const Tensor& b, double rtol = 1e-9) {
        Tensor ad = a.to(Device.cpu).astype(Type.Double).contiguous();
        Tensor bd = b.to(Device.cpu).astype(Type.Double).contiguous();
        if (ad.shape() != bd.shape()) return false;
        for (cytnx_uint64 i = 0; i < ad.storage().size(); ++i) {
          double x = ad.storage().at<double>(i), y = bd.storage().at<double>(i);
          if (std::abs(x - y) > rtol * (1.0 + std::abs(y))) return false;
        }
        return true;
      }

      TEST(SvdTruncate, GpuFlagCombinations) {
        for (auto dtype : {Type.Double, Type.ComplexDouble}) {
          Tensor T = Tensor({6, 5}, dtype).to(Device.cuda);
          InitTensorUniform(T, /*seed=*/dtype);
          const cytnx_uint64 keep = 3;

          // NOTE: the GPU cuSvd backend does not support is_UvT=false (it crashes when U/vT are not
          // computed, see issue #831), so we only exercise is_UvT=true here. This still verifies
          // that S/U/vT are truncated correctly through the refactored cudaMemcpyTruncation.
          std::vector<Tensor> ref = linalg::Svd_truncate(T, keep, 0., /*is_UvT=*/true, 0, 1);
          ASSERT_EQ(ref.size(), 3u);
          const cytnx_uint64 k = ref[0].shape()[0];
          EXPECT_EQ(k, keep);
          EXPECT_EQ(ref[1].shape(), std::vector<cytnx_uint64>({6, k}));
          EXPECT_EQ(ref[2].shape(), std::vector<cytnx_uint64>({k, 5}));
        }
      }

      TEST(GesvdTruncate, GpuFlagCombinations) {
        for (auto dtype : {Type.Double, Type.ComplexDouble}) {
          Tensor T = Tensor({6, 5}, dtype).to(Device.cuda);
          InitTensorUniform(T, /*seed=*/dtype);
          const cytnx_uint64 keep = 3;

          std::vector<Tensor> ref = linalg::Gesvd_truncate(T, keep, 0., true, true, 0, 1);
          ASSERT_EQ(ref.size(), 3u);
          const cytnx_uint64 k = ref[0].shape()[0];
          EXPECT_EQ(k, keep);

          for (bool is_U : {false, true}) {
            for (bool is_vT : {false, true}) {
              std::vector<Tensor> out = linalg::Gesvd_truncate(T, keep, 0., is_U, is_vT, 0, 1);
              const cytnx_uint64 expected = 1u + (is_U ? 1u : 0u) + (is_vT ? 1u : 0u);
              ASSERT_EQ(out.size(), expected) << "is_U=" << is_U << " is_vT=" << is_vT;
              EXPECT_TRUE(SingularValsClose(out[0], ref[0]))
                << "S mismatch, is_U=" << is_U << " is_vT=" << is_vT;
              cytnx_uint64 idx = 1;
              if (is_U) {
                EXPECT_EQ(out[idx].shape(), std::vector<cytnx_uint64>({6, k}));
                ++idx;
              }
              if (is_vT) {
                EXPECT_EQ(out[idx].shape(), std::vector<cytnx_uint64>({k, 5}));
              }
            }
          }
        }
      }

      TEST(Rsvd, GpuFlagCombinations) {
        const unsigned int seed = 42;
        const cytnx_uint64 keep = 3;
        // Use oversampling large enough that samplenum >= n_singlu, i.e. the full-SVD path (no
        // random projection).
        const cytnx_uint64 summand = 10;
        const double factor = 1.;
        const cytnx_uint64 power_it = 2;

        for (auto dtype : {Type.Double, Type.ComplexDouble}) {
          Tensor T = Tensor({8, 6}, dtype).to(Device.cuda);
          InitTensorUniform(T, /*seed=*/3);

          std::vector<Tensor> ref =
            linalg::Rsvd(T, keep, 0., true, true, 0, 1, summand, factor, power_it, seed);
          ASSERT_EQ(ref.size(), 3u);
          const cytnx_uint64 k = ref[0].shape()[0];

          for (bool is_U : {false, true}) {
            for (bool is_vT : {false, true}) {
              std::vector<Tensor> out =
                linalg::Rsvd(T, keep, 0., is_U, is_vT, 0, 1, summand, factor, power_it, seed);
              const cytnx_uint64 expected = 1u + (is_U ? 1u : 0u) + (is_vT ? 1u : 0u);
              ASSERT_EQ(out.size(), expected) << "is_U=" << is_U << " is_vT=" << is_vT;
              EXPECT_TRUE(SingularValsClose(out[0], ref[0], 1e-8))
                << "S mismatch, is_U=" << is_U << " is_vT=" << is_vT;
              cytnx_uint64 idx = 1;
              if (is_U) {
                EXPECT_EQ(out[idx].shape()[1], k);  // U truncated to k columns
                ++idx;
              }
              if (is_vT) {
                EXPECT_EQ(out[idx].shape()[0], k);  // vT truncated to k rows
              }
            }
          }
        }
      }

      TEST(SvdTruncate, GpuNoTruncationReturnsZeroError) {
        Tensor T = Tensor({6, 5}, Type.Double).to(Device.cuda);
        InitTensorUniform(T, 7);
        const cytnx_uint64 full = 5;  // min(6, 5) singular values
        for (unsigned int re : {1u, 2u}) {
          std::vector<Tensor> out = linalg::Svd_truncate(T, full, 0., true, re, 1);
          ASSERT_EQ(out.size(), 4u) << "[S, U, vT, terr], return_err=" << re;
          EXPECT_EQ(out[0].shape()[0], full);
          Tensor terr = out.back().to(Device.cpu);
          if (re == 1) {
            EXPECT_TRUE(terr.is_scalar());
            EXPECT_DOUBLE_EQ(terr.storage().at<double>(0), 0.0) << "return_err=" << re;
          } else {
            EXPECT_EQ(terr.shape(), std::vector<cytnx_uint64>({0}));
            EXPECT_TRUE(terr.is_empty()) << "return_err=" << re;
          }
        }
      }

      // A huge err would truncate almost everything; mindim must floor the kept count. On the
      // cuQuantum backend this exercises the "restart with keepdim=mindim" path in Gesvd_truncate.
      TEST(GesvdTruncate, GpuMindimFloor) {
        Tensor T = Tensor({6, 5}, Type.Double).to(Device.cuda);
        InitTensorUniform(T, 9);
        const cytnx_uint64 keepdim = 5, mindim = 3;
        std::vector<Tensor> out =
          linalg::Gesvd_truncate(T, keepdim, /*err=*/1e9, true, true, 0, mindim);
        ASSERT_EQ(out.size(), 3u);  // [S, U, vT]
        EXPECT_EQ(out[0].shape()[0], mindim);  // kept count floored at mindim despite huge err
        EXPECT_EQ(out[1].shape(), std::vector<cytnx_uint64>({6, mindim}));
        EXPECT_EQ(out[2].shape(), std::vector<cytnx_uint64>({mindim, 5}));
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
