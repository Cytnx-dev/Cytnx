#include "BlockFermionicUniTensor_test.h"

#include <algorithm>
#include <cmath>

namespace cytnx {
  namespace test {

    /*=====test info=====
    describe:scalar product between two vectors
    ====================*/
    TEST_F(BlockFermionicUniTensorTest, VectorContract) {
      // 1+2*2-3*3-4*4-5*5-6*6+7*7+8*8 = 32
      UniTensor out = BFUT1.contract(BFUT2);
      EXPECT_EQ(out.uten_type(), UTenType.BlockFermionic);
      EXPECT_EQ(out.rank(), 0);
      EXPECT_EQ(out.rowrank(), 0);
      EXPECT_TRUE(out.bonds().empty());
      EXPECT_TRUE(out.shape().empty());
      EXPECT_EQ(out.syms(), BFUT1.syms());
      EXPECT_EQ(out.signflip(), std::vector<bool>({false}));
      EXPECT_TRUE(out.get_block_().is_scalar());
      EXPECT_TRUE(std::abs(double(out.item().real()) - 32.0) < 1e-12);
      EXPECT_DOUBLE_EQ(out.item<cytnx_double>(), 32.0);
      EXPECT_THROW(BFUT1.item(), error);
    }

    TEST_F(BlockFermionicUniTensorTest, ZeroExtentBondProducesEmptyTensor) {
      const Bond empty = Bond(BD_IN, {{0}}, {0}, {Symmetry::FermionParity()});
      const Bond unit = Bond(BD_OUT, {{0}}, {1}, {Symmetry::FermionParity()});
      UniTensor tensor({empty, unit}, {"empty", "unit"}, 1, Type.Double, Device.cpu);

      EXPECT_EQ(tensor.uten_type(), UTenType.BlockFermionic);
      EXPECT_EQ(tensor.shape(), (std::vector<cytnx_uint64>{0, 1}));
      EXPECT_EQ(tensor.size(), 0);
      EXPECT_TRUE(tensor.is_empty());
      EXPECT_FALSE(tensor.is_void());
      EXPECT_FALSE(tensor.is_scalar());
      EXPECT_EQ(tensor.dtype(), Type.Double);
      EXPECT_EQ(tensor.device(), Device.cpu);
      ASSERT_EQ(tensor.Nblocks(), 1);
      EXPECT_TRUE(tensor.get_block_(0).is_empty());

      tensor.Save(temp_file_path);
      const UniTensor loaded = UniTensor::Load(temp_file_path);
      EXPECT_EQ(loaded.uten_type(), UTenType.BlockFermionic);
      EXPECT_EQ(loaded.shape(), tensor.shape());
      EXPECT_EQ(loaded.syms(), tensor.syms());
      EXPECT_TRUE(loaded.is_empty());
      ASSERT_EQ(loaded.Nblocks(), 1);
      EXPECT_TRUE(loaded.get_block_(0).is_empty());
    }

    /*=====test info=====
    describe:contraction
    ====================*/

    TEST_F(BlockFermionicUniTensorTest, SimpleTensorContract) {
      // 1+2*2-3*3-4*4-5*5-6*6+7*7+8*8 = 32
      EXPECT_TRUE(abs(BFUT3.contract(BFUT2).at({0, 0}) - 32) < 1e-13);
    }

    /*=====test info=====
    describe:contraction with mixed dtypes (double lhs, float rhs); exercises
             the #ifdef UNI_MKL dtype-cast path added with Gemm_Batch, including
             fermionic sign flips encoded in alpha
    ====================*/

    TEST_F(BlockFermionicUniTensorTest, ContractMixedDtype) {
      UniTensor L = BFUT1.astype(Type.Double);
      UniTensor R = BFUT2.astype(Type.Float);
      // 1+2*2-3*3-4*4-5*5-6*6+7*7+8*8 = 32 (same as VectorContract; verifies sign flip
      // preserved)
      UniTensor out = L.contract(R);
      EXPECT_EQ(out.uten_type(), UTenType.BlockFermionic);
      EXPECT_EQ(out.dtype(), Type.Double);
      EXPECT_EQ(out.rank(), 0);
      EXPECT_EQ(out.syms(), L.syms());
      EXPECT_EQ(out.signflip(), std::vector<bool>({false}));
      EXPECT_TRUE(std::abs(double(out.item().real()) - 32.0) < 1e-5);
    }

    TEST_F(BlockFermionicUniTensorTest, NormReturnsScalarTensor) {
      Tensor norm = BFUT1.Norm();
      EXPECT_TRUE(norm.is_scalar());
      EXPECT_GT(double(norm.item().real()), 0.0);
    }

    TEST_F(BlockFermionicUniTensorTest, TraceRankZeroScalarPreservesSymmetryMetadata) {
      Bond bi = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
      UniTensor bkf = UniTensor({bi, bi.redirect()}, {"a", "b"}, 1, Type.Double, Device.cpu);
      bkf.at({0, 0}) = 2.0;
      bkf.at({1, 1}) = 3.0;

      UniTensor traced = bkf.Trace("a", "b");
      EXPECT_EQ(traced.uten_type(), UTenType.BlockFermionic);
      EXPECT_EQ(traced.rank(), 0);
      EXPECT_EQ(traced.rowrank(), 0);
      EXPECT_TRUE(traced.bonds().empty());
      EXPECT_TRUE(traced.shape().empty());
      EXPECT_EQ(traced.syms(), bkf.syms());
      EXPECT_FALSE(traced.is_diag());
      EXPECT_EQ(traced.signflip(), std::vector<bool>({false}));
      EXPECT_TRUE(traced.get_block_().is_scalar());
      EXPECT_TRUE(traced.get_block_({}).is_scalar());
      EXPECT_DOUBLE_EQ(double(traced.at({}).real()), -1.0);
      EXPECT_NO_THROW(traced.to_dense());
      testing::internal::CaptureStdout();
      EXPECT_NO_THROW(traced.print_block(0, false));
      EXPECT_NE(testing::internal::GetCapturedStdout().find("rank-0 scalar block"),
                std::string::npos);

      UniTensor loaded_scalar;
      traced.Save(temp_file_path);
      EXPECT_NO_THROW(loaded_scalar = UniTensor::Load(temp_file_path));
      EXPECT_EQ(loaded_scalar.uten_type(), UTenType.BlockFermionic);
      EXPECT_EQ(loaded_scalar.rank(), 0);
      EXPECT_EQ(loaded_scalar.syms(), bkf.syms());
      EXPECT_EQ(loaded_scalar.signflip(), std::vector<bool>({false}));
      EXPECT_DOUBLE_EQ(double(loaded_scalar.at({}).real()), -1.0);

      UniTensor same_sym_sum = traced.clone();
      EXPECT_NO_THROW(same_sym_sum += loaded_scalar);
      EXPECT_DOUBLE_EQ(double(same_sym_sum.at({}).real()), -2.0);

      Bond fnum_bond = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionNumber()});
      UniTensor fnum =
        UniTensor({fnum_bond, fnum_bond.redirect()}, {"a", "b"}, 1, Type.Double, Device.cpu);
      fnum.at({0, 0}) = 2.0;
      fnum.at({1, 1}) = 3.0;
      UniTensor fnum_scalar = fnum.Trace("a", "b");
      EXPECT_EQ(fnum_scalar.uten_type(), UTenType.BlockFermionic);
      EXPECT_EQ(fnum_scalar.rank(), 0);
      EXPECT_NE(fnum_scalar.syms(), traced.syms());
      EXPECT_THROW(
        {
          UniTensor ignored = traced + fnum_scalar;
          (void)ignored;
        },
        std::logic_error);
      EXPECT_THROW(
        {
          UniTensor ignored = traced - fnum_scalar;
          (void)ignored;
        },
        std::logic_error);
      EXPECT_THROW(
        {
          UniTensor ignored = traced * fnum_scalar;
          (void)ignored;
        },
        std::logic_error);
      EXPECT_THROW(
        {
          UniTensor ignored = traced / fnum_scalar;
          (void)ignored;
        },
        std::logic_error);

      UniTensor scalar_contract;
      EXPECT_NO_THROW(scalar_contract = traced.contract(loaded_scalar));
      EXPECT_EQ(scalar_contract.uten_type(), UTenType.BlockFermionic);
      EXPECT_EQ(scalar_contract.rank(), 0);
      EXPECT_EQ(scalar_contract.syms(), bkf.syms());
      EXPECT_EQ(scalar_contract.signflip(), std::vector<bool>({false}));
      EXPECT_DOUBLE_EQ(double(scalar_contract.at({}).real()), 1.0);

      UniTensor traced_inplace = bkf.clone();
      traced_inplace.Trace_("a", "b");
      EXPECT_EQ(traced_inplace.uten_type(), UTenType.BlockFermionic);
      EXPECT_EQ(traced_inplace.rank(), 0);
      EXPECT_EQ(traced_inplace.rowrank(), 0);
      EXPECT_TRUE(traced_inplace.bonds().empty());
      EXPECT_TRUE(traced_inplace.shape().empty());
      EXPECT_EQ(traced_inplace.syms(), bkf.syms());
      EXPECT_FALSE(traced_inplace.is_diag());
      EXPECT_EQ(traced_inplace.signflip(), std::vector<bool>({false}));
      EXPECT_TRUE(traced_inplace.get_block_().is_scalar());
      EXPECT_DOUBLE_EQ(double(traced_inplace.at({}).real()), -1.0);
      EXPECT_NO_THROW(traced_inplace.to_dense());

      UniTensor transposed = traced_inplace.Transpose();
      EXPECT_EQ(transposed.uten_type(), UTenType.BlockFermionic);
      EXPECT_EQ(transposed.rank(), 0);
      EXPECT_EQ(transposed.rowrank(), 0);
      EXPECT_TRUE(transposed.bonds().empty());
      EXPECT_TRUE(transposed.shape().empty());
      EXPECT_EQ(transposed.syms(), bkf.syms());
      EXPECT_FALSE(transposed.is_diag());
      EXPECT_EQ(transposed.signflip(), std::vector<bool>({false}));
      EXPECT_TRUE(transposed.get_block_().is_scalar());
      EXPECT_DOUBLE_EQ(double(transposed.at({}).real()), -1.0);

      traced_inplace.Transpose_();
      EXPECT_EQ(traced_inplace.uten_type(), UTenType.BlockFermionic);
      EXPECT_EQ(traced_inplace.rank(), 0);
      EXPECT_EQ(traced_inplace.rowrank(), 0);
      EXPECT_TRUE(traced_inplace.bonds().empty());
      EXPECT_TRUE(traced_inplace.shape().empty());
      EXPECT_EQ(traced_inplace.syms(), bkf.syms());
      EXPECT_FALSE(traced_inplace.is_diag());
      EXPECT_EQ(traced_inplace.signflip(), std::vector<bool>({false}));
      EXPECT_TRUE(traced_inplace.get_block_().is_scalar());
      EXPECT_DOUBLE_EQ(double(traced_inplace.at({}).real()), -1.0);

      UniTensor diag = UniTensor({bi, bi.redirect()}, {"a", "b"}, 1, Type.Double, Device.cpu, true);
      diag.get_block_(0).fill(2.0);
      diag.get_block_(1).fill(3.0);
      UniTensor traced_diag = diag.Trace("a", "b");
      EXPECT_EQ(traced_diag.uten_type(), UTenType.BlockFermionic);
      EXPECT_EQ(traced_diag.rank(), 0);
      EXPECT_FALSE(traced_diag.is_diag());
      EXPECT_EQ(traced_diag.syms(), diag.syms());
      EXPECT_NO_THROW(traced_diag.to_dense());
    }

    TEST_F(BlockFermionicUniTensorTest, DiagonalPermutePreservesBlocks) {
      Bond bi = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
      UniTensor diag = UniTensor({bi, bi.redirect()}, {"a", "b"}, 1, Type.Double, Device.cpu, true);
      diag.get_block_(0).fill(2.0);
      diag.get_block_(1).fill(3.0);

      UniTensor permuted = diag.permute(std::vector<std::string>{"b", "a"});
      EXPECT_TRUE(permuted.is_diag());
      EXPECT_FALSE(permuted.get_block_(0).is_void());
      EXPECT_FALSE(permuted.get_block_(1).is_void());
      EXPECT_DOUBLE_EQ(permuted.get_block_(0).item<cytnx_double>(), 2.0);
      EXPECT_DOUBLE_EQ(permuted.get_block_(1).item<cytnx_double>(), 3.0);

      UniTensor no_signflip = diag.permute_nosignflip({"b", "a"});
      EXPECT_TRUE(no_signflip.is_diag());
      EXPECT_FALSE(no_signflip.get_block_(0).is_void());
      EXPECT_FALSE(no_signflip.get_block_(1).is_void());
      EXPECT_DOUBLE_EQ(no_signflip.get_block_(0).item<cytnx_double>(), 2.0);
      EXPECT_DOUBLE_EQ(no_signflip.get_block_(1).item<cytnx_double>(), 3.0);
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
      // check power, only works for unpermuted tensor (otherwise, sign structure of T*T
      // and T.Pow(2.) differs)
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

    TEST_F(BlockFermionicUniTensorTest, GroupBasis) {
      auto out = BFUT4.group_basis();

      // careful: the quantum numbers are sorted; the last index such changes from
      // (0,1,2) to (1,2,0)
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
      const char* fname = temp_file_path.c_str();
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
      // BFUT1: rank=3, rowrank=2, bonds=[BD_IN(a), BD_IN(b), BD_OUT(c)],
      // shape=(2,2,4)
      EXPECT_EQ(BFUT1.rowrank(), 2);

      auto tmp = BFUT1.Transpose();

      // rowrank must be rank - old_rowrank = 3 - 2 = 1
      EXPECT_EQ(tmp.rowrank(), 1);
      EXPECT_EQ(tmp.rank(), 3);

      // index order is reversed: new [0,1,2] = old [c,b,a]
      EXPECT_EQ(tmp.labels()[0], "c");
      EXPECT_EQ(tmp.labels()[1], "b");
      EXPECT_EQ(tmp.labels()[2], "a");

      // bonds are redirected: old BD_OUT(c)->BD_IN, old BD_IN(b)->BD_OUT,
      // old BD_IN(a)->BD_OUT
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

    /*=====test info=====
    describe:regression test for issue #724 on the BlockFermionicUniTensor
    path. Two UniTensors sharing the same underlying block Tensors (via
             relabel(), documented to share data with the original) must not
             corrupt each other's metadata when one of them is permuted in
             place with permute_(). BlockFermionicUniTensor::permute_ is a
             distinct implementation from BlockUniTensor's (it additionally
             updates the per-block sign-flip state), so it gets its own
    test.
    ====================*/

    TEST_F(BlockFermionicUniTensorTest, PermuteInPlaceOnSharedBlockDoesNotCorruptOtherHolder) {
      UniTensor uT = BFUT3.clone().set_name("uT");
      UniTensor uT2 = uT.relabel({"p", "q", "r", "s", "t"}).set_name("uT2");

      // Precondition: the two UniTensors really do share the same block
      // storage.
      ASSERT_TRUE(uT.same_data(uT2));
      ASSERT_EQ(uT.Nblocks(), uT2.Nblocks());

      const auto orig_shape = uT.shape();
      const auto orig_labels = uT.labels();
      const auto orig_signflip = uT.signflip();
      // The 8 non-zero entries of BFUT3 as initialized in the fixture.
      const std::vector<std::vector<cytnx_uint64>> nz_locs = {
        {0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 2, 0, 0}, {0, 1, 3, 0, 0},
        {1, 0, 2, 0, 0}, {1, 0, 3, 0, 0}, {1, 1, 0, 0, 0}, {1, 1, 1, 0, 0}};
      const std::vector<double> nz_vals = {1., 2., 3., 4., 5., 6., 7., 8.};

      std::vector<cytnx_int64> a = {3, 1, 4, 2, 0};
      uT2.permute_(a);

      // uT2 changed as expected: after resolving the pending sign flips
      // it must match the BFUT3PERM reference (same permutation as the
      // pre-existing LinAlgElementwise test).
      EXPECT_TRUE(AreEqUniTensor(BFUT3PERM, uT2.contiguous().apply()));

      // uT must be completely unaffected: shape, labels, sign-flip
      // state, and data all preserved.
      ASSERT_EQ(uT.shape(), orig_shape);
      EXPECT_EQ(uT.labels(), orig_labels);
      EXPECT_EQ(uT.signflip(), orig_signflip);
      EXPECT_TRUE(AreEqUniTensor(uT, BFUT3));
      // Reading uT after uT2's in-place permute must not even throw
      // (stale block/qnum mapping vs. a physically permuted shared
      // block Tensor can manifest as an out-of-bound access).
      for (size_t n = 0; n < nz_locs.size(); n++) {
        try {
          ASSERT_TRUE(uT.at(nz_locs[n]).exists());
          EXPECT_DOUBLE_EQ(double(uT.at(nz_locs[n]).real()), nz_vals[n]);
        } catch (const std::exception& e) {
          ADD_FAILURE() << "uT.at(nz_locs[" << n
                        << "]) threw after uT2.permute_() "
                           "(shared-block metadata corrupted): "
                        << e.what();
        }
      }
    }

    // ============ convert_from / from_ ============

    // A rank-3 fermionic tensor is permuted (which flips signs on some
    // but not all blocks), converted to Dense and back, then permuted
    // to the original leg order; the round-trip reproduces the
    // original. The converted tensor carries one sign flag per block.
    TEST_F(BlockFermionicUniTensorTest, ConvertFromPermuteRoundtrip) {
      const double tol = 1e-12;
      UniTensor Tp = BFUT1.permute({2, 0, 1});
      ASSERT_GT(Tp.Nblocks(),
                (cytnx_uint64)Tp.rank());  // more blocks than rank

      // the permutation puts sign flips on some (not all) blocks
      bool any_flip = false, any_noflip = false;
      for (bool b : Tp.signflip()) {
        any_flip = any_flip || b;
        any_noflip = any_noflip || !b;
      }
      ASSERT_TRUE(any_flip);
      ASSERT_TRUE(any_noflip);

      UniTensor D = UniTensor(zeros(Tp.shape()));
      D.convert_from(Tp);  // BlockFermionic -> Dense (resolves the
                           // pending sign flips)

      UniTensor BKF2 = Tp.clone();
      BKF2.convert_from(D);  // Dense -> BlockFermionic
      EXPECT_EQ(BKF2.signflip().size(),
                BKF2.Nblocks());  // one sign flag per block

      UniTensor back = BKF2.permute(BFUT1.labels());
      EXPECT_TRUE(AreNearlyEqUniTensor(back.apply(), BFUT1.apply(), tol));
    }

    // Dense -> BlockFermionic honors tol: a nonzero symmetry-forbidden
    // entry is rejected at the default tol=0, but force=true / large
    // tol drop it and the allowed entries reproduce the original
    // exactly.
    TEST_F(BlockFermionicUniTensorTest, ConvertFromTolForbiddenNonzero) {
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
      Bt.convert_from(D, false,
                      10.0);  // large tol tolerates it, same result
      EXPECT_TRUE(AreEqUniTensor(Bt, BKF));
    }

    // Converting a BlockFermionic into a diagonal Dense is not
    // supported and must throw.
    TEST_F(BlockFermionicUniTensorTest, ConvertFromDiagonalDenseTargetThrows) {
      Bond bi = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
      UniTensor BKF = UniTensor({bi, bi.redirect()});

      UniTensor Ddiag = UniTensor(zeros({2}), true);  // diagonal Dense, shape (2,2)
      EXPECT_ANY_THROW(Ddiag.convert_from(BKF));
    }

    // ============ to_dense / to_dense_ ============

    // A diagonal BlockFermionicUniTensor is expanded to a full one:
    // each rank-1 (diagonal) block becomes a diagonal matrix and
    // is_diag() becomes false. A twist flips the sign of the odd-parity
    // block, and those per-block sign flags are carried over unchanged.
    TEST_F(BlockFermionicUniTensorTest, ToDenseDiag) {
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
        EXPECT_EQ(dense.signflip(),
                  B.signflip());  // sign flags carried over unchanged
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

    // to_dense on an already non-diagonal BlockFermionicUniTensor is a
    // no-op: the tensor, including its per-block sign flags (made
    // non-trivial here by an initial permutation), is returned
    // unchanged.
    TEST_F(BlockFermionicUniTensorTest, ToDenseNonDiag) {
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

    /*=====test info=====
    describe:Integer-dtype fermionic block contractions must not throw
    in MKL builds. Gemm_Batch rejects dtype > 4; the Matmul fallback
    must be taken instead.
    ====================*/

    TEST_F(BlockFermionicUniTensorTest, ContractIntegerDtype) {
      Bond bi = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 2}, {Symmetry::FermionParity()});
      UniTensor L = UniTensor({bi, bi.redirect()}, {"a", "b"}, 1, Type.Int64, Device.cpu, false);
      UniTensor R = UniTensor({bi, bi.redirect()}, {"b", "c"}, 1, Type.Int64, Device.cpu, false);
      L.at({0, 0}) = 1;
      L.at({2, 2}) = 2;
      R.at({0, 0}) = 3;
      R.at({2, 2}) = 4;
      UniTensor out;
      EXPECT_NO_THROW(out = Contract(L, R));
      EXPECT_EQ(int64_t(out.at({0, 0}).real()), 3);
      EXPECT_EQ(int64_t(out.at({2, 2}).real()), 8);
    }

    /*=====test info=====
    describe:regression test for issue #724 on the fermionic
    contract() path. contract() used to permute_/reshape_ the
    operands' blocks in place and restore them afterward. When the
    two operands alias each other's blocks (relabel() shares block
    storage), the in-place mutation of the left operand corrupts the
    right operand mid- contraction: the contraction throws or
    produces wrong values. contract() must treat both operands'
    blocks as read-only.
    ====================*/

    TEST_F(BlockFermionicUniTensorTest, ContractAliasedSharedBlocksOperandsIntact) {
      Bond ba = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 2}, {Symmetry::FermionParity()});
      Bond bb = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 3}, {Symmetry::FermionParity()});
      Bond bc = Bond(BD_OUT, {Qs(0) >> 1, Qs(1) >> 2}, {Symmetry::FermionParity()});
      UniTensor A = UniTensor({ba, bb, bc}, {"a", "b", "c"});
      random::uniform_(A, -1.0, 1.0, 42);
      // B shares A's blocks; contracting over "a" and "c" makes
      // the two operands of contract() alias each other's
      // blocks (including block-with-itself pairs).
      UniTensor B = A.relabel({"c", "d", "a"});
      ASSERT_TRUE(A.same_data(B));

      UniTensor Asnap = A.clone();  // pristine copy of the shared data
      // reference result from fully independent operands
      UniTensor expected = Contract(Asnap.clone(), Asnap.clone().relabel({"c", "d", "a"}));

      UniTensor got;
      try {
        got = Contract(A, B);
      } catch (const std::exception& e) {
        FAIL() << "Contract() on operands sharing blocks threw: " << e.what();
      }

      ASSERT_EQ(got.Nblocks(), expected.Nblocks());
      EXPECT_EQ(got.signflip(), expected.signflip());
      for (cytnx_uint64 i = 0; i < got.Nblocks(); i++) {
        EXPECT_TRUE(AreNearlyEqTensor(got.get_blocks_()[i], expected.get_blocks_()[i], 1e-12));
      }

      // both operands must be intact: values, shapes, signflip,
      // and contiguity
      ASSERT_EQ(A.Nblocks(), Asnap.Nblocks());
      EXPECT_EQ(A.signflip(), Asnap.signflip());
      for (cytnx_uint64 i = 0; i < A.Nblocks(); i++) {
        EXPECT_EQ(A.get_blocks_()[i].shape(), Asnap.get_blocks_()[i].shape());
        EXPECT_TRUE(AreNearlyEqTensor(A.get_blocks_()[i], Asnap.get_blocks_()[i], 0.0));
      }
      EXPECT_TRUE(A.is_contiguous());
      EXPECT_TRUE(B.is_contiguous());
    }

    /*=====test info=====
    describe:regression test for issue #724 on the fermionic
    contract() path. A third UniTensor sharing blocks with an
    operand (via relabel()) must not observe any change from the
    contraction. The pre-fix mutate- and-restore left the shared
    blocks with replaced, permuted storage (non-contiguous),
    even though the values were restored.
    ====================*/

    TEST_F(BlockFermionicUniTensorTest, ContractLeavesSharedBlockObserverUntouched) {
      Bond ba = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 2}, {Symmetry::FermionParity()});
      Bond bb = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 3}, {Symmetry::FermionParity()});
      Bond bc = Bond(BD_OUT, {Qs(0) >> 1, Qs(1) >> 2}, {Symmetry::FermionParity()});
      UniTensor A = UniTensor({ba, bb, bc}, {"a", "b", "c"});
      random::uniform_(A, -1.0, 1.0, 7);
      UniTensor observer = A.relabel({"x", "y", "z"});  // shares A's blocks; not an operand
      ASSERT_TRUE(A.same_data(observer));
      UniTensor snap = A.clone();

      UniTensor R = A.clone().relabel({"c", "d", "a"});  // independent right operand
      UniTensor got = Contract(A, R);
      (void)got;

      ASSERT_EQ(observer.Nblocks(), snap.Nblocks());
      EXPECT_EQ(observer.signflip(), snap.signflip());
      for (cytnx_uint64 i = 0; i < observer.Nblocks(); i++) {
        EXPECT_EQ(observer.get_blocks_()[i].shape(), snap.get_blocks_()[i].shape());
        EXPECT_TRUE(AreNearlyEqTensor(observer.get_blocks_()[i], snap.get_blocks_()[i], 0.0));
      }
      EXPECT_TRUE(observer.is_contiguous());
      EXPECT_TRUE(R.is_contiguous());
    }

    /*=====test info=====
    describe:#1052 scope note -- BlockFermionicUniTensor::combineBonds(vector<cytnx_int64>, force)
      starts with an unconditional cytnx_error_msg(true, "not implemented yet."), so every public
      entry point that reaches it (combineBond(labels), combineBonds(labels),
      combineBonds(indices, force, by_label)) always throws before any bond-combining logic runs.
      That makes the two overlapping-memcpy sites fixed alongside BlockUniTensor's (the cb_stride
      shift and the _inner_to_outer_idx tail shift, a few lines below this guard) unreachable code
      today -- there is no way to exercise them through the public API without also lifting this
      "not implemented" guard, which is a functional change to fermionic sign-flip handling (see
      the function's own "TODOfermion: signflips need to be included!!!" comment) outside this
      memory-safety-only PR. Pin the current throw so this scope statement stays true; the memcpy
      fix itself is applied for source hygiene/consistency with BlockUniTensor and to avoid the
      same latent bug when this function is eventually implemented.
    ====================*/
    TEST_F(BlockFermionicUniTensorTest, CombineBondsStillUnimplemented) {
      Bond ba = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 2}, {Symmetry::FermionParity()});
      Bond bb = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 3}, {Symmetry::FermionParity()});
      Bond bc = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
      Bond bd = Bond(BD_OUT, {Qs(0) >> 1, Qs(1) >> 2}, {Symmetry::FermionParity()});
      UniTensor A = UniTensor({ba, bb, bc, bd}, {"a", "b", "c", "d"}, 3, Type.Double);
      EXPECT_THROW(A.combineBond_({"a", "b"}, /*force=*/true), cytnx::error);
      EXPECT_THROW(A.combineBond_({"a", "b", "c"}, /*force=*/true), cytnx::error);
    }

  }  // namespace test
}  // namespace cytnx
