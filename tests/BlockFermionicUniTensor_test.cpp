#include "BlockFermionicUniTensor_test.h"

TEST_F(BlockFermionicUniTensorTest, VectorContract) {
  // 1+2*2-3*3-4*4-5*5-6*6+7*7+8*8 = 32
  EXPECT_TRUE(abs(BFUT1.contract(BFUT2).item() - 32) < 1e-12);
}

TEST_F(BlockFermionicUniTensorTest, SimpleTensorContract) {
  // 1+2*2-3*3-4*4-5*5-6*6+7*7+8*8 = 32
  EXPECT_TRUE(abs(BFUT3.contract(BFUT2).at({0, 0}) - 32) < 1e-13);
}

TEST_F(BlockFermionicUniTensorTest, LinAlgElementwise) {
  const double tol = 1e-14;
  UniTensor T = BFUT3.permute({3, 1, 4, 2, 0}).contiguous();
  EXPECT_TRUE(AreEqUniTensor(BFUT3PERM, T.apply()));
  UniTensor res = T + T;
  EXPECT_TRUE(AreNearlyEqUniTensor(2. * BFUT3PERM, res.apply_(), tol));
  res = T + BFUT3PERM;
  EXPECT_TRUE(AreNearlyEqUniTensor(2. * BFUT3PERM, res.apply_(), tol));
  res = (T + T + T + T) / 4.;
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), BFUT3PERM, tol));
  res = (T + T + BFUT3PERM + T) / 4.;
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), BFUT3PERM, tol));
  res = (2 * T) - T;
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), BFUT3PERM, tol));
  res = (2 * T) - BFUT3PERM;
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), BFUT3PERM, tol));
  res = BFUT3PERM * BFUT3PERM;
  UniTensor ref = T * T;
  res.permute_(ref.labels());
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), ref.apply_(), tol));
  res = (T * T * T) / T;
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), BFUT3PERM * BFUT3PERM, tol));
  // negation
  res = BFUT3PERM * (-1. * BFUT3PERM);
  ref = -1 * ref;
  ref.permute_(res.labels());
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), ref.apply_(), tol));
  res = (-1. * BFUT3PERM) * BFUT3PERM;
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), ref, tol));
  // commutative property
  res = BFUT3PERM.permute_nosignflip(T.labels()) * T;
  ref = T * BFUT3PERM.permute_nosignflip(T.labels());
  ref.permute_(res.labels());
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), ref.apply_(), tol));
  res = BFUT3PERM.permute_nosignflip(T.labels()).permute_(BFUT3PERM.labels()) * BFUT3PERM;
  ref = BFUT3PERM * BFUT3PERM.permute_nosignflip(T.labels()).permute_(BFUT3PERM.labels());
  ref.permute_(res.labels());
  EXPECT_TRUE(AreNearlyEqUniTensor(res.apply_(), ref.apply_(), tol));
  // test inline
  res = T;
  res += T;
  res /= 2.;
  EXPECT_EQ(AreNearlyEqUniTensor(res.apply_(), T.apply(), tol), true);
  // test Mul and Div for tensors
  res = T.clone();
  res *= T;
  ref = T * T;
  EXPECT_EQ(AreNearlyEqUniTensor(res.apply_(), ref.apply_(), tol), true);
  res /= T;
  EXPECT_EQ(AreNearlyEqUniTensor(res.apply_(), T.apply(), tol), true);
  // check power, only works for unpermuted tensor (otherwise, sign structure of T*T and T.Pow(2.)
  // differs)
  res = BFUT3.clone();
  res *= BFUT3;
  EXPECT_EQ(AreNearlyEqUniTensor(res.apply_(), BFUT3.Pow(2.).apply(), tol), true);
}

TEST_F(BlockFermionicUniTensorTest, group_basis) {
  auto out = BFUT4.group_basis();

  // careful: the quantum numbers are sorted; the last index such changes from (0,1,2) to (1,2,0)
  EXPECT_DOUBLE_EQ(double(out.at({0, 0, 0}).real()), double(1));
  EXPECT_DOUBLE_EQ(double(out.at({0, 0, 1}).real()), double(2));
  EXPECT_DOUBLE_EQ(double(out.at({0, 1, 2}).real()), double(3));
  EXPECT_DOUBLE_EQ(double(out.at({1, 0, 2}).real()), double(4));
  EXPECT_DOUBLE_EQ(double(out.at({2, 0, 2}).real()), double(5));
  EXPECT_DOUBLE_EQ(double(out.at({1, 1, 0}).real()), double(6));
  EXPECT_DOUBLE_EQ(double(out.at({2, 1, 0}).real()), double(7));
  EXPECT_DOUBLE_EQ(double(out.at({1, 1, 1}).real()), double(8));
  EXPECT_DOUBLE_EQ(double(out.at({2, 1, 1}).real()), double(9));

  out = BFUT4.permute({1, 0, 2});
  out = out.group_basis();
  out.permute_(BFUT4.labels());
  out.apply_();
  // the block indices can differ, therefore I compare the elements
  EXPECT_DOUBLE_EQ(double(out.at({0, 0, 0}).real()), double(1));
  EXPECT_DOUBLE_EQ(double(out.at({0, 0, 1}).real()), double(2));
  EXPECT_DOUBLE_EQ(double(out.at({0, 1, 2}).real()), double(3));
  EXPECT_DOUBLE_EQ(double(out.at({1, 0, 2}).real()), double(4));
  EXPECT_DOUBLE_EQ(double(out.at({2, 0, 2}).real()), double(5));
  EXPECT_DOUBLE_EQ(double(out.at({1, 1, 0}).real()), double(6));
  EXPECT_DOUBLE_EQ(double(out.at({2, 1, 0}).real()), double(7));
  EXPECT_DOUBLE_EQ(double(out.at({1, 1, 1}).real()), double(8));
  EXPECT_DOUBLE_EQ(double(out.at({2, 1, 1}).real()), double(9));

  out = BFUT4.permute({1, 2, 0});
  out = out.group_basis();
  out.permute_(BFUT4.labels());
  out.apply_();
  // the block indices can differ, therefore I compare the elements
  EXPECT_DOUBLE_EQ(double(out.at({0, 0, 0}).real()), double(1));
  EXPECT_DOUBLE_EQ(double(out.at({0, 0, 1}).real()), double(2));
  EXPECT_DOUBLE_EQ(double(out.at({0, 1, 2}).real()), double(3));
  EXPECT_DOUBLE_EQ(double(out.at({1, 0, 2}).real()), double(4));
  EXPECT_DOUBLE_EQ(double(out.at({2, 0, 2}).real()), double(5));
  EXPECT_DOUBLE_EQ(double(out.at({1, 1, 0}).real()), double(6));
  EXPECT_DOUBLE_EQ(double(out.at({2, 1, 0}).real()), double(7));
  EXPECT_DOUBLE_EQ(double(out.at({1, 1, 1}).real()), double(8));
  EXPECT_DOUBLE_EQ(double(out.at({2, 1, 1}).real()), double(9));
}

TEST_F(BlockFermionicUniTensorTest, SaveLoad) {
  BFUT1.Save("BFUT1");
  UniTensor BFUTloaded = BFUTloaded.Load("BFUT1.cytnx");
  EXPECT_TRUE(AreEqUniTensor(BFUT1, BFUTloaded));
}
