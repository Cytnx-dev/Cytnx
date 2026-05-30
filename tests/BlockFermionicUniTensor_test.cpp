#include "BlockFermionicUniTensor_test.h"

#include <algorithm>
#include <cmath>

/*=====test info=====
describe:scalar product between two vectors
====================*/
TEST_F(BlockFermionicUniTensorTest, VectorContract) {
  // 1+2*2-3*3-4*4-5*5-6*6+7*7+8*8 = 32
  EXPECT_TRUE(abs(BFUT1.contract(BFUT2).item() - 32) < 1e-12);
}

/*=====test info=====
describe:contraction
====================*/
TEST_F(BlockFermionicUniTensorTest, SimpleTensorContract) {
  // 1+2*2-3*3-4*4-5*5-6*6+7*7+8*8 = 32
  EXPECT_TRUE(abs(BFUT3.contract(BFUT2).at({0, 0}) - 32) < 1e-13);
}

/*=====test info=====
describe:some elementwise linear algebra functions
====================*/
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

/*=====test info=====
describe:test fermion_twists behavior on tagged fermionic tensors
====================*/
TEST_F(BlockFermionicUniTensorTest, FermionTwists) {
  UniTensor twisted = BFUT5.fermion_twists();
  UniTensor manual = BFUT5.clone();
  for (cytnx_int64 idx = manual.rowrank(); idx < manual.rank(); idx++) {
    if (manual.bonds()[idx].type() != BD_BRA) manual.twist_(idx);
  }
  EXPECT_EQ(twisted.signflip(), manual.signflip());
  EXPECT_TRUE(AreEqUniTensor(twisted.apply_(), manual.apply_()));

  // applying fermion_twists_ twice toggles the same set of signs twice
  UniTensor twice = BFUT5.clone();
  twice.fermion_twists_().fermion_twists_();
  EXPECT_EQ(twice.signflip(), BFUT5.signflip());
  EXPECT_TRUE(AreEqUniTensor(twice.apply_(), BFUT5.apply()));
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

/*=====test info=====
describe:write to disc
====================*/
TEST_F(BlockFermionicUniTensorTest, SaveLoad) {
  BFUT1.Save(temp_file_path);
  UniTensor BFUT1_loaded = BFUT1_loaded.Load(temp_file_path);
  EXPECT_TRUE(AreEqUniTensor(BFUT1, BFUT1_loaded));
  // for char*
  const char *fname = temp_file_path.c_str();
  BFUT1.Save(fname);
  UniTensor BFUT1_loaded_char_save = BFUT1_loaded_char_save.Load(temp_file_path);
  EXPECT_TRUE(AreEqUniTensor(BFUT1, BFUT1_loaded_char_save));
  UniTensor BFUT1_loaded_char_load = BFUT1_loaded_char_load.Load(fname);
  EXPECT_TRUE(AreEqUniTensor(BFUT1, BFUT1_loaded_char_load));
}

/*=====test info=====
describe:test Transpose and Transpose_ for BlockFermionicUniTensor:
  rowrank is updated, index order is reversed, bonds are redirected,
  and element values are preserved without sign flips.
====================*/
TEST_F(BlockFermionicUniTensorTest, Transpose) {
  // BFUT1: rank=3, rowrank=2, bonds=[BD_IN(a), BD_IN(b), BD_OUT(c)], shape=(2,2,4)
  EXPECT_EQ(BFUT1.rowrank(), 2);

  auto tmp = BFUT1.Transpose();

  // rowrank must be rank - old_rowrank = 3 - 2 = 1
  EXPECT_EQ(tmp.rowrank(), 1);
  EXPECT_EQ(tmp.rank(), 3);

  // index order is reversed: new [0,1,2] = old [c,b,a]
  EXPECT_EQ(tmp.labels()[0], "c");
  EXPECT_EQ(tmp.labels()[1], "b");
  EXPECT_EQ(tmp.labels()[2], "a");

  // bonds are redirected: old BD_OUT(c)->BD_IN, old BD_IN(b)->BD_OUT, old BD_IN(a)->BD_OUT
  EXPECT_EQ(tmp.bonds()[0].type(), BD_IN);
  EXPECT_EQ(tmp.bonds()[1].type(), BD_OUT);
  EXPECT_EQ(tmp.bonds()[2].type(), BD_OUT);

  // element at old {a,b,c} appears at new {c,b,a}; no sign flips
  EXPECT_DOUBLE_EQ(double(tmp.at({0, 0, 0}).real()), 1.);
  EXPECT_DOUBLE_EQ(double(tmp.at({1, 0, 0}).real()), 2.);
  EXPECT_DOUBLE_EQ(double(tmp.at({2, 1, 0}).real()), 3.);
  EXPECT_DOUBLE_EQ(double(tmp.at({3, 1, 0}).real()), 4.);
  EXPECT_DOUBLE_EQ(double(tmp.at({2, 0, 1}).real()), 5.);
  EXPECT_DOUBLE_EQ(double(tmp.at({3, 0, 1}).real()), 6.);
  EXPECT_DOUBLE_EQ(double(tmp.at({0, 1, 1}).real()), 7.);
  EXPECT_DOUBLE_EQ(double(tmp.at({1, 1, 1}).real()), 8.);

  // Transpose is an involution: T.Transpose().Transpose() == T
  EXPECT_TRUE(AreEqUniTensor(tmp.Transpose(), BFUT1));

  // in-place version must match
  auto tmp2 = BFUT1.clone();
  tmp2.Transpose_();
  EXPECT_EQ(tmp2.rowrank(), 1);
  EXPECT_EQ(tmp2.bonds()[0].type(), BD_IN);
  EXPECT_EQ(tmp2.bonds()[1].type(), BD_OUT);
  EXPECT_EQ(tmp2.bonds()[2].type(), BD_OUT);
  EXPECT_TRUE(AreEqUniTensor(tmp2, tmp));
}
