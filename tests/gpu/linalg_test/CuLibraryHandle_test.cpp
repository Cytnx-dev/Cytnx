#include <cstddef>

#include <gtest/gtest.h>

#include "backend/utils_internal_gpu/cuLibraryHandle_gpu.hpp"
#include "cytnx.hpp"
#include "gpu_test_tools.h"

using namespace cytnx;

// #1144: cuBLAS and cuSOLVER handles were created and destroyed around every call, at 52 live
// sites. They are now cached per device, so repeated calls must hand back the identical handle --
// this is the direct assertion of the change, and it does not depend on timing or on memory
// thresholds.
TEST(CuLibraryHandle, CublasHandleIsSharedAcrossCalls) {
  const cublasHandle_t first = utils_internal::get_cublas_handle();
  EXPECT_NE(first, nullptr);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(utils_internal::get_cublas_handle(), first);
  }
}

TEST(CuLibraryHandle, CusolverDnHandleIsSharedAcrossCalls) {
  const cusolverDnHandle_t first = utils_internal::get_cusolverdn_handle();
  EXPECT_NE(first, nullptr);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(utils_internal::get_cusolverdn_handle(), first);
  }
}

// The behaviour change is that many operations now run through one long-lived handle instead of a
// fresh one each time. These check results against values computed by hand -- not against another
// Cytnx path, which could share the same fault -- and interleave the operations so that each one
// runs on a handle already used by the others.
TEST(CuLibraryHandle, SharedHandleGivesCorrectResultsWhenInterleaved) {
  // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]  =>  A*B = [[19, 22], [43, 50]]
  Tensor a = zeros({2, 2}, Type.Double);
  Tensor b = zeros({2, 2}, Type.Double);
  const double av[4] = {1, 2, 3, 4};
  const double bv[4] = {5, 6, 7, 8};
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      a.at({i, j}) = av[i * 2 + j];
      b.at({i, j}) = bv[i * 2 + j];
    }
  }
  const Tensor ga = a.to(Device.cuda);
  const Tensor gb = b.to(Device.cuda);

  // v = [1, 2, 3], w = [4, 5, 6]  =>  v.w = 32;  n = [3, 4]  =>  ||n|| = 5
  Tensor v = zeros({3}, Type.Double), w = zeros({3}, Type.Double), n = zeros({2}, Type.Double);
  const double vv[3] = {1, 2, 3};
  const double wv[3] = {4, 5, 6};
  for (std::size_t i = 0; i < 3; ++i) {
    v.at({i}) = vv[i];
    w.at({i}) = wv[i];
  }
  n.at({0}) = 3.0;
  n.at({1}) = 4.0;
  const Tensor gv = v.to(Device.cuda), gw = w.to(Device.cuda), gn = n.to(Device.cuda);

  const double kExpectedAb[4] = {19, 22, 43, 50};
  Tensor expected_ab = zeros({2, 2}, Type.Double);
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j) expected_ab.at({i, j}) = kExpectedAb[i * 2 + j];

  // Three rounds so every operation runs on a handle already used by the other two.
  for (int round = 0; round < 3; ++round) {
    const Tensor ab = linalg::Matmul(ga, gb).to(Device.cpu);
    EXPECT_TRUE(gpu_test::AreNearlyEqTensor(ab, expected_ab, 1e-12)) << "round " << round;

    // Vectordot, Norm and Det all return rank-0 tensors (Init({}, ...)), so the scalar
    // accessor is required -- at({0}) would be a rank mismatch.
    const Tensor dot = linalg::Vectordot(gv, gw).to(Device.cpu);
    EXPECT_NEAR(dot.item<cytnx_double>(), 32.0, 1e-12) << "round " << round;

    const Tensor norm = linalg::Norm(gn).to(Device.cpu);
    EXPECT_NEAR(norm.item<cytnx_double>(), 5.0, 1e-12) << "round " << round;

    // det [[1, 2], [3, 4]] = 1*4 - 2*3 = -2  (cuSOLVER path, shared cusolverDn handle)
    const Tensor det = linalg::Det(ga).to(Device.cpu);
    EXPECT_NEAR(det.item<cytnx_double>(), -2.0, 1e-12) << "round " << round;
  }
}
