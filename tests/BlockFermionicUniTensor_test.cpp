#include "BlockFermionicUniTensor_test.h"

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
  BFUT1.Save(temp_file_path);
  UniTensor BFUT1_loaded = BFUT1_loaded.Load(temp_file_path + ".cytnx");
  EXPECT_TRUE(AreEqUniTensor(BFUT1, BFUT1_loaded));
  // for char*
  const char *fname = temp_file_path.c_str();
  BFUT1.Save(fname);
  UniTensor BFUT1_loaded_char_save = BFUT1_loaded_char_save.Load(temp_file_path + ".cytnx");
  EXPECT_TRUE(AreEqUniTensor(BFUT1, BFUT1_loaded_char_save));
  UniTensor BFUT1_loaded_char_load = BFUT1_loaded_char_load.Load(fname);
  EXPECT_TRUE(AreEqUniTensor(BFUT1, BFUT1_loaded_char_load));
}
