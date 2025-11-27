#include "BlockFermionicUniTensor_test.h"

TEST_F(BlockFermionicUniTensorTest, VectorContract) {
  // 1+2*2-3*3-4*4-5*5-6*6+7*7+8*8 = 32
  EXPECT_EQ(abs(BFUT1.contract(BFUT2).item() - 32) < 1e-12, true);
}

TEST_F(BlockFermionicUniTensorTest, SimpleTensorContract) {
  // 1+2*2-3*3-4*4-5*5-6*6+7*7+8*8 = 32
  EXPECT_EQ(abs(BFUT3.contract(BFUT2).at({0, 0}) - 32) < 1e-13, true);
}

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
  EXPECT_EQ(AreNearlyEqUniTensor((T * T * T) / T, BFUT3PERM * BFUT3PERM, tol), true);
  // test inline
  UniTensor T2 = T;
  T2 += T;
  T2 /= 2.;
  EXPECT_EQ(AreNearlyEqUniTensor(T2, T, tol), true);
  // test Mul and Div for tensors
  UniTensor Tsq = T.clone();
  Tsq *= T;
  EXPECT_EQ(AreNearlyEqUniTensor(Tsq, T * T, tol), true);
  Tsq /= T;
  EXPECT_EQ(AreNearlyEqUniTensor(Tsq, T, tol), true);
  // check power, only works for unpermuted tensor (otherwise, sign structure of T*T and T.Pow(2.)
  // differs)
  Tsq = BFUT3.clone();
  Tsq *= BFUT3;
  EXPECT_EQ(AreNearlyEqUniTensor(Tsq, BFUT3.Pow(2.), tol), true);
}

TEST_F(BlockFermionicUniTensorTest, SaveLoad) {
  BFUT1.Save("BFUT1");
  UniTensor BFUTloaded = BFUTloaded.Load("BFUT1.cytnx");
  EXPECT_EQ(AreEqUniTensor(BFUT1, BFUTloaded), true);
}
