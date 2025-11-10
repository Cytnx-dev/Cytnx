#include "BlockFermionicUniTensor_test.h"

/*=====test info=====
describe:scalar product between two vectors
====================*/
TEST_F(BlockFermionicUniTensorTest, VectorContract) {
  // 1+2*2-3*3-4*4-5*5-6*6+7*7+8*8 = 32
  EXPECT_EQ(abs(BFUT1.contract(BFUT2).item() - 32) < 1e-12, true);
}

/*=====test info=====
describe:contraction
====================*/
TEST_F(BlockFermionicUniTensorTest, SimpleTensorContract) {
  // 1+2*2-3*3-4*4-5*5-6*6+7*7+8*8 = 32
  EXPECT_EQ(abs(BFUT3.contract(BFUT2).at({0, 0}) - 32) < 1e-13, true);
}

/*=====test info=====
describe:some elementwise linear algebra functions
====================*/
TEST_F(BlockFermionicUniTensorTest, LinAlgElementwise) {
  const double tol = 1e-14;
  UniTensor T = BFUT3.permute({3, 1, 4, 2, 0}).contiguous();
  EXPECT_EQ(AreEqUniTensor(BFUT3PERM, T), true);
  EXPECT_EQ(AreNearlyEqUniTensor(2. * BFUT3PERM, T + T, tol), true);
  EXPECT_EQ(AreNearlyEqUniTensor(2. * BFUT3PERM, T + BFUT3PERM, tol), true);
  EXPECT_EQ(AreNearlyEqUniTensor((T + T + T + T) / 4., BFUT3PERM, tol), true);
  EXPECT_EQ(AreNearlyEqUniTensor((T + T + BFUT3PERM + T) / 4., BFUT3PERM, tol), true);
  EXPECT_EQ(AreNearlyEqUniTensor((2 * T) - T, BFUT3PERM, tol), true);
  EXPECT_EQ(AreNearlyEqUniTensor((2 * T) - BFUT3PERM, BFUT3PERM, tol), true);
}

/*=====test info=====
describe:test pseudo-inverse
====================*/
TEST_F(BlockFermionicUniTensorTest, Inv) {
  const double tol = 1e-14;
  double clip = 1e-15;
  EXPECT_TRUE(AreNearlyEqUniTensor(BFUT3.Inv(clip), BFUT3INV, tol));
  UniTensor T = BFUT3.permute({3, 1, 4, 2, 0}).contiguous();
  EXPECT_TRUE(AreNearlyEqUniTensor(T.Inv(clip).Inv_(clip), T, tol));
  EXPECT_FALSE(AreNearlyEqUniTensor(T.Inv(clip), T, tol));
  clip = 0.1;  // test actual clipping as well
  auto tmp = T.clone();
  tmp.Inv_(clip);  // test inline version
  EXPECT_TRUE(AreEqUniTensor(T.Inv(clip), tmp));
  tmp = T.clone();
  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 4; k++)
        for (size_t l = 0; l < 2; l++)
          for (size_t m = 0; m < 2; m++) {
            // elements are permuted!
            auto proxy = tmp.at({l, j, m, k, i});
            if (proxy.exists()) {
              Scalar val = proxy;
              if (val.abs() <= clip)
                proxy = 0.;
              else
                proxy = 1. / proxy;
            }
          }
  EXPECT_TRUE(AreNearlyEqUniTensor(T.Inv(clip), tmp, tol));
}

/*=====test info=====
describe:test power
====================*/
TEST_F(BlockFermionicUniTensorTest, Pow) {
  const double tol = 1e-14;
  UniTensor T = BFUT3.permute({3, 1, 4, 2, 0}).contiguous();
  // only even powers are products, odd powers differ by signflips
  EXPECT_TRUE(AreNearlyEqUniTensor(T.Pow(3.), T * T * T, tol));
  auto tmp = T.clone();
  tmp.Pow_(2.3);  // test inline version
  EXPECT_TRUE(AreEqUniTensor(T.Pow(2.3), tmp));
  for (double p = 0.; p < 1.6; p += 0.5) {
    tmp = T.clone();
    for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 2; j++)
        for (size_t k = 0; k < 4; k++)
          for (size_t l = 0; l < 2; l++)
            for (size_t m = 0; m < 2; m++) {
              // elements are permuted!
              auto proxy = tmp.at({l, j, m, k, i});
              if (proxy.exists()) {
                Scalar val = proxy;
                proxy = std::pow((cytnx_double)val, p);
              }
            }
    EXPECT_TRUE(AreNearlyEqUniTensor(T.Pow(p), tmp, tol));
  }
}

/*=====test info=====
describe:write to disc
====================*/
TEST_F(BlockFermionicUniTensorTest, SaveLoad) {
  BFUT1.Save("BFUT1");
  UniTensor BFUTloaded = BFUTloaded.Load("BFUT1.cytnx");
  EXPECT_EQ(AreEqUniTensor(BFUT1, BFUTloaded), true);
}
