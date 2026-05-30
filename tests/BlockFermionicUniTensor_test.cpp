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

// ============ convert_from / from_ ============

// A rank-3 fermionic tensor is permuted (which flips signs on some but not all blocks), converted
// to Dense and back, then permuted to the original leg order; the round-trip reproduces the
// original. The converted tensor carries one sign flag per block.
TEST_F(BlockFermionicUniTensorTest, convert_from_permute_roundtrip) {
  const double tol = 1e-12;
  UniTensor Tp = BFUT1.permute({2, 0, 1});
  ASSERT_GT(Tp.Nblocks(), (cytnx_uint64)Tp.rank());  // more blocks than rank

  // the permutation puts sign flips on some (not all) blocks
  bool any_flip = false, any_noflip = false;
  for (bool b : Tp.signflip()) {
    any_flip = any_flip || b;
    any_noflip = any_noflip || !b;
  }
  ASSERT_TRUE(any_flip);
  ASSERT_TRUE(any_noflip);

  UniTensor D = UniTensor(zeros(Tp.shape()));
  D.convert_from(Tp);  // BlockFermionic -> Dense (resolves the pending sign flips)

  UniTensor BKF2 = Tp.clone();
  BKF2.convert_from(D);  // Dense -> BlockFermionic
  EXPECT_EQ(BKF2.signflip().size(), BKF2.Nblocks());  // one sign flag per block

  UniTensor back = BKF2.permute(BFUT1.labels());
  EXPECT_TRUE(AreNearlyEqUniTensor(back.apply(), BFUT1.apply(), tol));
}

// Dense -> BlockFermionic honors tol: a nonzero symmetry-forbidden entry is rejected at the default
// tol=0, but force=true / large tol drop it and the allowed entries reproduce the original exactly.
TEST_F(BlockFermionicUniTensorTest, convert_from_tol_forbidden_nonzero) {
  Bond bi = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
  UniTensor BKF = UniTensor({bi, bi.redirect()});
  BKF.at({0, 0}) = 1.0;  // even-even block
  BKF.at({1, 1}) = 2.0;  // odd-odd block

  UniTensor D = UniTensor(zeros(BKF.shape()));
  D.convert_from(BKF);
  D.at({0, 1}) = 7.0;  // forbidden sector (even row, odd col)

  UniTensor B0 = UniTensor({bi, bi.redirect()});
  EXPECT_ANY_THROW(B0.convert_from(D));  // tol defaults to 0 -> rejected

  UniTensor Bf = UniTensor({bi, bi.redirect()});
  Bf.convert_from(D, true);  // force drops the forbidden entry
  EXPECT_TRUE(AreEqUniTensor(Bf, BKF));

  UniTensor Bt = UniTensor({bi, bi.redirect()});
  Bt.convert_from(D, false, 10.0);  // large tol tolerates it, same result
  EXPECT_TRUE(AreEqUniTensor(Bt, BKF));
}

// Converting a BlockFermionic into a diagonal Dense is not supported and must throw.
TEST_F(BlockFermionicUniTensorTest, convert_from_diagonal_dense_target_throws) {
  Bond bi = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
  UniTensor BKF = UniTensor({bi, bi.redirect()});

  UniTensor Ddiag = UniTensor(zeros(2), true);  // diagonal Dense, shape (2,2)
  EXPECT_ANY_THROW(Ddiag.convert_from(BKF));
}

// ============ to_dense / to_dense_ ============

// A diagonal BlockFermionicUniTensor is expanded to a full one: each rank-1 (diagonal) block
// becomes a diagonal matrix and is_diag() becomes false. A twist flips the sign of the odd-parity
// block, and those per-block sign flags are carried over unchanged.
TEST_F(BlockFermionicUniTensorTest, to_dense_diag) {
  Bond bd = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 3}, {Symmetry::FermionParity()});
  for (auto dtype : {Type.ComplexDouble, Type.Double}) {
    UniTensor B = UniTensor({bd, bd.redirect()}, {"a", "b"}, 1, dtype, Device.cpu, true);
    random::uniform_(B, -10.0, 10.0, 0);
    B.twist_(0);  // sign flip on the odd-parity block

    bool any_flip = false, any_noflip = false;
    for (bool s : B.signflip()) {
      any_flip = any_flip || s;
      any_noflip = any_noflip || !s;
    }
    ASSERT_TRUE(any_flip);  // sign flips on some...
    ASSERT_TRUE(any_noflip);  // ...but not all blocks

    UniTensor dense = B.to_dense();
    EXPECT_TRUE(B.is_diag());
    EXPECT_FALSE(dense.is_diag());
    EXPECT_EQ(dense.dtype(), dtype);
    EXPECT_EQ(dense.signflip(), B.signflip());  // sign flags carried over unchanged
    ASSERT_EQ(dense.Nblocks(), B.Nblocks());
    for (cytnx_uint64 b = 0; b < B.Nblocks(); b++)
      EXPECT_TRUE(AreNearlyEqTensor(dense.get_block_(b), linalg::Diag(B.get_block_(b)), 1e-14));

    UniTensor Bp = B.clone();
    Bp.to_dense_();
    EXPECT_FALSE(Bp.is_diag());
    EXPECT_EQ(Bp.signflip(), B.signflip());
    EXPECT_TRUE(AreEqUniTensor(Bp, dense));
  }
}

// to_dense on an already non-diagonal BlockFermionicUniTensor is a no-op: the tensor, including its
// per-block sign flags (made non-trivial here by an initial permutation), is returned unchanged.
TEST_F(BlockFermionicUniTensorTest, to_dense_non_diag) {
  UniTensor T = BFUT5.permute({2, 0, 3, 1});  // sign flips on some blocks
  bool any_flip = false;
  for (bool s : T.signflip()) any_flip = any_flip || s;
  ASSERT_TRUE(any_flip);
  EXPECT_FALSE(T.is_diag());

  UniTensor dense = T.to_dense();
  EXPECT_TRUE(AreEqUniTensor(T, dense));
  EXPECT_EQ(dense.signflip(), T.signflip());

  UniTensor Tp = T.clone();
  Tp.to_dense_();
  EXPECT_TRUE(AreEqUniTensor(T, Tp));
  EXPECT_EQ(Tp.signflip(), T.signflip());
}
