#include <cmath>

#include "gtest/gtest.h"

#include "cytnx.hpp"

using namespace cytnx;
using namespace cytnx::tn_algo;

namespace {

  // Contract a rank-3 MPS site tensor A (shape [left, phys, right], rowrank 2)
  // with its own conjugate over the (left, phys) indices. For a left-orthonormal
  // tensor the result is the identity on the (right, right') bond.
  static UniTensor LeftGram(const UniTensor &A) {
    auto a = A.relabel({"L", "P", "R"});
    auto ac = A.Conj().relabel({"L", "P", "Rp"});
    return Contract(a, ac);  // -> {R, Rp}
  }

  // Contract a rank-3 MPS site tensor A over the (phys, right) indices with its
  // conjugate. For a right-orthonormal tensor the result is the identity on the
  // (left, left') bond.
  static UniTensor RightGram(const UniTensor &A) {
    auto a = A.relabel({"L", "P", "R"});
    auto ac = A.Conj().relabel({"Lp", "P", "R"});
    return Contract(a, ac);  // -> {L, Lp}
  }

  static bool IsIdentity(const UniTensor &G, double tol = 1e-9) {
    auto blk = G.get_block_();
    auto shape = blk.shape();
    if (shape.size() != 2 || shape[0] != shape[1]) return false;
    cytnx_uint64 D = shape[0];
    for (cytnx_uint64 i = 0; i < D; i++) {
      for (cytnx_uint64 j = 0; j < D; j++) {
        double expected = (i == j) ? 1.0 : 0.0;
        if (std::abs(blk.at<double>({i, j}) - expected) > tol) return false;
      }
    }
    return true;
  }

  // Regression test for the construction crash reported in #920: building a
  // regular MPS ran Into_Lortho(), whose chained Svd calls reused the fixed
  // "_aux_L"/"_aux_R" bond labels and aborted with a duplicated-label error.
  TEST(MPS, RegularConstructionDoesNotCrash) {
    EXPECT_NO_THROW({ MPS mps(4, 2, 8); });
    EXPECT_NO_THROW({ MPS mps(8, 2, 16); });
    EXPECT_NO_THROW({ MPS mps(6, 3, 5); });
  }

  TEST(MPS, RegularConstructionMetadata) {
    MPS mps(5, 2, 8);
    EXPECT_EQ(mps.size(), 5u);
    EXPECT_EQ(mps.mps_type(), 0);  // RegularMPS
    EXPECT_EQ(mps.virt_dim(), 8);
    for (cytnx_uint64 i = 0; i < mps.size(); i++) {
      EXPECT_EQ(mps.data()[i].rank(), 3u);
      EXPECT_EQ(mps.phys_dim(i), 2);
    }
    // Open boundary conditions: outermost virtual bonds are trivial.
    EXPECT_EQ(mps.data().front().shape()[0], 1u);
    EXPECT_EQ(mps.data().back().shape()[2], 1u);
  }

  // After Init(), the MPS is left-orthogonalized with the orthogonality center
  // swept off the right edge, so every site tensor must be left-orthonormal.
  TEST(MPS, ConstructionIsLeftOrthonormal) {
    MPS mps(6, 2, 8);
    EXPECT_EQ(mps.S_loc(), static_cast<cytnx_int64>(mps.size()));
    for (cytnx_uint64 p = 0; p < mps.size(); p++) {
      EXPECT_TRUE(IsIdentity(LeftGram(mps.data()[p])))
        << "site " << p << " is not left-orthonormal";
    }
  }

  // The final Svd in Into_Lortho() discards the singular values of the right
  // edge, so a freshly constructed MPS is normalized to one.
  TEST(MPS, ConstructionIsNormalized) {
    MPS mps(6, 2, 8);
    double n = double(mps.norm());
    EXPECT_NEAR(n, 1.0, 1e-9);
  }

  TEST(MPS, NonUniformPhysDim) {
    std::vector<cytnx_uint64> phys = {2, 3, 2, 4};
    MPS mps;
    EXPECT_NO_THROW({ mps.Init(4, phys, 10); });
    ASSERT_EQ(mps.size(), 4u);
    for (cytnx_uint64 i = 0; i < phys.size(); i++) {
      EXPECT_EQ(mps.phys_dim(i), static_cast<cytnx_int64>(phys[i]));
      EXPECT_TRUE(IsIdentity(LeftGram(mps.data()[i])))
        << "site " << i << " is not left-orthonormal";
    }
    EXPECT_NEAR(double(mps.norm()), 1.0, 1e-9);
  }

  // Init_Msector also funnels through Into_Lortho() and was equally affected by
  // the #920 label collision.
  TEST(MPS, InitMsectorDoesNotCrash) {
    MPS mps;
    std::vector<cytnx_uint64> phys = {2, 2, 2, 2};
    std::vector<cytnx_int64> select = {0, 1, 0, 1};
    EXPECT_NO_THROW({ mps.Init_Msector(4, phys, 8, select); });
    EXPECT_EQ(mps.size(), 4u);
    for (cytnx_uint64 p = 0; p < mps.size(); p++) {
      EXPECT_TRUE(IsIdentity(LeftGram(mps.data()[p])))
        << "site " << p << " is not left-orthonormal";
    }
    EXPECT_NEAR(double(mps.norm()), 1.0, 1e-9);
  }

  // Sweeping the orthogonality center is a gauge transformation: it must leave
  // the represented state (and hence its norm) unchanged, and it must not hit
  // the same chained-Svd label collision as Into_Lortho().
  TEST(MPS, OrthoCenterSweepPreservesNorm) {
    MPS mps(6, 2, 8);
    double n0 = double(mps.norm());
    ASSERT_NEAR(n0, 1.0, 1e-9);

    // Sweep the center from the right edge all the way to the left edge.
    EXPECT_NO_THROW({
      while (mps.S_loc() > -1) mps.S_mvleft();
    });
    EXPECT_NEAR(double(mps.norm()), 1.0, 1e-9);

    // ... and back to the right edge again.
    EXPECT_NO_THROW({
      while (mps.S_loc() < static_cast<cytnx_int64>(mps.size())) mps.S_mvright();
    });
    EXPECT_NEAR(double(mps.norm()), 1.0, 1e-9);
  }

  // After fully sweeping the center to the left edge, every site tensor must be
  // right-orthonormal.
  TEST(MPS, SweepLeftIsRightOrthonormal) {
    MPS mps(6, 2, 8);
    while (mps.S_loc() > -1) mps.S_mvleft();
    EXPECT_EQ(mps.S_loc(), -1);
    for (cytnx_uint64 p = 0; p < mps.size(); p++) {
      EXPECT_TRUE(IsIdentity(RightGram(mps.data()[p])))
        << "site " << p << " is not right-orthonormal";
    }
  }

  TEST(MPS, CloneIsIndependent) {
    MPS mps(4, 2, 8);
    MPS cp = mps.clone();
    ASSERT_EQ(cp.size(), mps.size());
    // Mutating the clone's center must not change the original.
    double n_before = double(mps.norm());
    cp.S_mvleft();
    EXPECT_EQ(mps.S_loc(), static_cast<cytnx_int64>(mps.size()));
    EXPECT_NEAR(double(mps.norm()), n_before, 1e-12);
  }

}  // namespace
