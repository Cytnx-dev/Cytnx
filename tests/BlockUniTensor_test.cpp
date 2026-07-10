#include "BlockUniTensor_test.h"

TEST_F(BlockUniTensorTest, Init_by_Tensor) {
  // not a valid operation
  EXPECT_ANY_THROW(BkUt.Init_by_Tensor(tzero345, false, -1));
}

TEST_F(BlockUniTensorTest, Init) {
  // different types
  EXPECT_NO_THROW(BkUt.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.Float, Device.cpu,
                            false, false));
  EXPECT_NO_THROW(BkUt.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.Double, Device.cpu,
                            false, false));
  EXPECT_NO_THROW(BkUt.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.ComplexFloat,
                            Device.cpu, false, false));
  EXPECT_NO_THROW(BkUt.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.ComplexDouble,
                            Device.cpu, false, false));

  // on gpu device
  //  EXPECT_NO_THROW(BkUt.Init({phy,phy.redirect(),aux},{"a", "b",
  //  "c"},1,Type.Float,Device.cuda,false,false));
  //  EXPECT_NO_THROW(BkUt.Init({phy,phy.redirect(),aux},{"a", "b",
  //  "c"},1,Type.Double,Device.cuda,false,false));
  //  EXPECT_NO_THROW(BkUt.Init({phy,phy.redirect(),aux},{"a", "b",
  //  "c"},1,Type.ComplexFloat,Device.cuda,false,false));
  //  EXPECT_NO_THROW(BkUt.Init({phy,phy.redirect(),aux},{"a", "b",
  //  "c"},1,Type.ComplexDouble,Device.cuda,false,false));

  // valid rowranks
  EXPECT_ANY_THROW(BkUt.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 99, Type.Float,
                             Device.cpu, false, false));
  EXPECT_NO_THROW(BkUt.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 2, Type.Float, Device.cpu,
                            false, false));
  EXPECT_NO_THROW(BkUt.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.Float, Device.cpu,
                            false, false));
  EXPECT_NO_THROW(BkUt.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, -1, Type.Float, Device.cpu,
                            false, false));
  EXPECT_ANY_THROW(BkUt.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, -2, Type.Float,
                             Device.cpu, false, false));
  EXPECT_NO_THROW(BkUt.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 0, Type.Float, Device.cpu,
                            false, false));

  // is_diag = true, but rank>2
  EXPECT_ANY_THROW(
    BkUt.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.Float, Device.cpu, true, false));

  // is_diag = true, but rowrank!=1
  EXPECT_ANY_THROW(
    BkUt.Init({phy, phy.redirect()}, {"a", "b"}, 2, Type.Float, Device.cpu, true, false));

  // is_diag = true, but no outward bond
  EXPECT_ANY_THROW(BkUt.Init({phy, phy}, {"a", "b"}, 1, Type.Float, Device.cpu, true, false));
}

TEST_F(BlockUniTensorTest, set_rowrank) {
  // Spf is a rank-3 tensor
  EXPECT_ANY_THROW(Spf.set_rowrank(-2));  // set_rowrank cannot be negative!
  EXPECT_ANY_THROW(Spf.set_rowrank(-1));
  EXPECT_NO_THROW(Spf.set_rowrank(0));
  EXPECT_NO_THROW(Spf.set_rowrank(1));
  EXPECT_NO_THROW(Spf.set_rowrank(2));
  EXPECT_NO_THROW(Spf.set_rowrank(3));
  EXPECT_ANY_THROW(Spf.set_rowrank(4));  // set_rowrank can only from 0-3 for rank-3 tn
}

TEST_F(BlockUniTensorTest, Nblocks) { EXPECT_EQ(UT_pB.Nblocks(), 4); }

TEST_F(BlockUniTensorTest, dtype) {
  EXPECT_EQ(Spf.dtype(), Type.Float);
  EXPECT_EQ(Spd.dtype(), Type.Double);
  EXPECT_EQ(Spcf.dtype(), Type.ComplexFloat);
  EXPECT_EQ(Spcd.dtype(), Type.ComplexDouble);
}

TEST_F(BlockUniTensorTest, device) { EXPECT_EQ(Spf.device(), Device.cpu); }

TEST_F(BlockUniTensorTest, dtype_str) {
  EXPECT_EQ(Spf.dtype_str(), "Float (Float32)");
  EXPECT_EQ(Spd.dtype_str(), "Double (Float64)");
  EXPECT_EQ(Spcf.dtype_str(), "Complex Float (Complex Float32)");
  EXPECT_EQ(Spcd.dtype_str(), "Complex Double (Complex Float64)");
}

TEST_F(BlockUniTensorTest, device_str) { EXPECT_EQ(Spf.device_str(), "cytnx device: CPU"); }

TEST_F(BlockUniTensorTest, syms) {
  EXPECT_EQ(BUT1.syms(), std::vector<Symmetry>{Symmetry::U1()});
  EXPECT_EQ(BUT2.syms(), std::vector<Symmetry>({Symmetry::U1(), Symmetry::U1()}));
  EXPECT_EQ(BUT3.syms(), std::vector<Symmetry>({Symmetry::Zn(2), Symmetry::U1()}));
}

// Regression: UniTensor whose first bond is BD_BRA with a non-uniform symmetry
// list (here {U1, Zn(2)}) used to feed `syms[0].reverse_rule(...)` to every
// component `i` in `_fx_get_total_fluxs`. For `i = 1` that meant U1's reverse
// rule was applied to a Zn(2) qnum, producing a negative value that was then
// passed back into `syms[1].combine_rule` (Zn(2)). The result was silently
// wrong before strict Zn validation and a hard `std::logic_error` after it.
TEST_F(BlockUniTensorTest, FxGetTotalFluxsUsesPerComponentSymmetry) {
  std::vector<Symmetry> syms = {Symmetry::U1(), Symmetry::Zn(2)};
  Bond bd_bra = Bond(BD_BRA, {{0, 0}, {0, 1}, {1, 0}, {1, 1}}, {1, 1, 1, 1}, syms);
  Bond bd_ket = Bond(BD_KET, {{0, 0}, {0, 1}, {1, 0}, {1, 1}}, {1, 1, 1, 1}, syms);

  EXPECT_NO_THROW({
    UniTensor ut = UniTensor({bd_bra, bd_ket});
    EXPECT_EQ(ut.syms(), syms);
  });
}

TEST_F(BlockUniTensorTest, is_contiguous) {
  EXPECT_EQ(Spf.is_contiguous(), true);
  auto Spf_new = Spf.permute({2, 1, 0}, 1);
  EXPECT_EQ(Spf_new.is_contiguous(), false);
}

TEST_F(BlockUniTensorTest, shape) {
  EXPECT_EQ(std::vector<cytnx::cytnx_uint64>({2, 2, 1}), Spf.shape());
}
TEST_F(BlockUniTensorTest, is_blockform) {
  EXPECT_EQ(Spf.is_blockform(), true);
  EXPECT_EQ(utzero345.is_blockform(), false);
}
TEST_F(BlockUniTensorTest, clone) {
  UniTensor cloned = UT_pB_ans.clone();
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 9; j++)
      for (cytnx_uint64 k = 1; k < 30; k++) {
        EXPECT_EQ(cloned.at({i, j, k}).exists(), UT_pB_ans.at({i, j, k}).exists());
        if (cloned.at({i, j, k}).exists()) EXPECT_EQ(cloned.at({i, j, k}), UT_pB_ans.at({i, j, k}));
      }
}

TEST_F(BlockUniTensorTest, relabels) {
  BUT1 = BUT1.relabels({"a", "b", "cd", "d"});
  EXPECT_EQ(BUT1.labels()[0], "a");
  EXPECT_EQ(BUT1.labels()[1], "b");
  EXPECT_EQ(BUT1.labels()[2], "cd");
  EXPECT_EQ(BUT1.labels()[3], "d");
  BUT1 = BUT1.relabels({"1", "-1", "2", "1000"});
  EXPECT_EQ(BUT1.labels()[0], "1");
  EXPECT_EQ(BUT1.labels()[1], "-1");
  EXPECT_EQ(BUT1.labels()[2], "2");
  EXPECT_EQ(BUT1.labels()[3], "1000");

  EXPECT_THROW(BUT1.relabels({"a", "a", "b", "c"}), std::logic_error);
  EXPECT_THROW(BUT1.relabels({"1", "1", "0", "-1"}), std::logic_error);
  EXPECT_THROW(BUT1.relabels({"a"}), std::logic_error);
  EXPECT_THROW(BUT1.relabels({"1", "2"}), std::logic_error);
  EXPECT_THROW(BUT1.relabels({"a", "b", "c", "d", "e"}), std::logic_error);
}
TEST_F(BlockUniTensorTest, relabels_) {
  BUT1.relabels_({"a", "b", "cd", "d"});
  EXPECT_EQ(BUT1.labels()[0], "a");
  EXPECT_EQ(BUT1.labels()[1], "b");
  EXPECT_EQ(BUT1.labels()[2], "cd");
  EXPECT_EQ(BUT1.labels()[3], "d");
  BUT1.relabels_({"1", "-1", "2", "1000"});
  EXPECT_EQ(BUT1.labels()[0], "1");
  EXPECT_EQ(BUT1.labels()[1], "-1");
  EXPECT_EQ(BUT1.labels()[2], "2");
  EXPECT_EQ(BUT1.labels()[3], "1000");
  EXPECT_THROW(BUT1.relabels_({"a", "a", "b", "c"}), std::logic_error);
  EXPECT_THROW(BUT1.relabels_({"1", "1", "0", "-1"}), std::logic_error);
  EXPECT_THROW(BUT1.relabels_({"a"}), std::logic_error);
  EXPECT_THROW(BUT1.relabels_({"1", "2"}), std::logic_error);
  EXPECT_THROW(BUT1.relabels_({"a", "b", "c", "d", "e"}), std::logic_error);
}

TEST_F(BlockUniTensorTest, relabel) {
  auto tmp = BUT1.clone();
  BUT1 = BUT1.relabel({"a", "b", "cd", "d"});
  EXPECT_EQ(BUT1.labels()[0], "a");
  EXPECT_EQ(BUT1.labels()[1], "b");
  EXPECT_EQ(BUT1.labels()[2], "cd");
  EXPECT_EQ(BUT1.labels()[3], "d");
  BUT1 = BUT1.relabel({"1", "-1", "2", "1000"});
  EXPECT_EQ(BUT1.labels()[0], "1");
  EXPECT_EQ(BUT1.labels()[1], "-1");
  EXPECT_EQ(BUT1.labels()[2], "2");
  EXPECT_EQ(BUT1.labels()[3], "1000");

  EXPECT_THROW(BUT1.relabel({"a", "a", "b", "c"}), std::logic_error);
  EXPECT_THROW(BUT1.relabel({"1", "1", "0", "-1"}), std::logic_error);
  EXPECT_THROW(BUT1.relabel({"a"}), std::logic_error);
  EXPECT_THROW(BUT1.relabel({"1", "2"}), std::logic_error);
  EXPECT_THROW(BUT1.relabel({"a", "b", "c", "d", "e"}), std::logic_error);

  BUT1 = tmp;
  BUT1 = BUT1.relabel("0", "a");
  BUT1 = BUT1.relabel("1", "b");
  BUT1 = BUT1.relabel("2", "d");
  BUT1 = BUT1.relabel("3", "de");

  BUT1 = BUT1.relabel("b", "ggg");

  EXPECT_EQ(BUT1.labels()[0], "a");
  EXPECT_EQ(BUT1.labels()[1], "ggg");
  EXPECT_EQ(BUT1.labels()[2], "d");
  EXPECT_EQ(BUT1.labels()[3], "de");

  BUT1 = BUT1.relabel(0, "ccc");
  EXPECT_EQ(BUT1.labels()[0], "ccc");

  BUT1 = BUT1.relabel(3, "-1");
  EXPECT_EQ(BUT1.labels()[3], "-1");

  BUT1 = BUT1.relabel(1, "-199922");
  EXPECT_EQ(BUT1.labels()[1], "-199922");

  BUT1 = BUT1.relabel("-1", "0");
  EXPECT_EQ(BUT1.labels()[3], "0");

  // BUT1.relabel(0,'a');
  // EXPECT_EQ(BUT1.labels()[0],"a");
  EXPECT_THROW(BUT1.relabel(5, "a"), std::logic_error);
  EXPECT_THROW(BUT1.relabel(-1, "a"), std::logic_error);
  EXPECT_THROW(BUT1.relabel(0, "a").relabel(1, "a"), std::logic_error);
  // BUT1.relabel(0,"a").relabel(1,"a");
  // EXPECT_THROW(BUT1.relabel("a","b"),std::logic_error);
  // EXPECT_THROW(BUT1.relabel(5,'a'),std::logic_error);
}
TEST_F(BlockUniTensorTest, relabel_) {
  auto tmp = BUT1.clone();
  BUT1.relabel_({"a", "b", "cd", "d"});
  EXPECT_EQ(BUT1.labels()[0], "a");
  EXPECT_EQ(BUT1.labels()[1], "b");
  EXPECT_EQ(BUT1.labels()[2], "cd");
  EXPECT_EQ(BUT1.labels()[3], "d");
  BUT1.relabel_({"1", "-1", "2", "1000"});
  EXPECT_EQ(BUT1.labels()[0], "1");
  EXPECT_EQ(BUT1.labels()[1], "-1");
  EXPECT_EQ(BUT1.labels()[2], "2");
  EXPECT_EQ(BUT1.labels()[3], "1000");
  EXPECT_THROW(BUT1.relabel_({"a", "a", "b", "c"}), std::logic_error);
  EXPECT_THROW(BUT1.relabel_({"1", "1", "0", "-1"}), std::logic_error);
  EXPECT_THROW(BUT1.relabel_({"a"}), std::logic_error);
  EXPECT_THROW(BUT1.relabel_({"1", "2"}), std::logic_error);
  EXPECT_THROW(BUT1.relabel_({"a", "b", "c", "d", "e"}), std::logic_error);

  BUT1 = tmp;
  BUT1.relabel_("0", "a");
  BUT1.relabel_("1", "b");
  BUT1.relabel_("2", "d");
  BUT1.relabel_("3", "de");
  BUT1.relabel_("b", "ggg");

  EXPECT_EQ(BUT1.labels()[0], "a");
  EXPECT_EQ(BUT1.labels()[1], "ggg");
  EXPECT_EQ(BUT1.labels()[2], "d");
  EXPECT_EQ(BUT1.labels()[3], "de");

  BUT1.relabel_(0, "ccc");
  EXPECT_EQ(BUT1.labels()[0], "ccc");

  BUT1.relabel_(3, "-1");
  EXPECT_EQ(BUT1.labels()[3], "-1");

  BUT1.relabel_(1, "-199922");
  EXPECT_EQ(BUT1.labels()[1], "-199922");

  BUT1.relabel_("-1", "0");
  EXPECT_EQ(BUT1.labels()[3], "0");
  EXPECT_THROW(BUT1.relabel_(5, "a"), std::logic_error);
  EXPECT_THROW(BUT1.relabel_(-1, "a"), std::logic_error);
  // EXPECT_THROW(BUT1.relabel(0,"a").relabel(1,"a"),std::logic_error);
}

TEST_F(BlockUniTensorTest, astype) {
  UniTensor Spf2d = Spf.astype(Type.Double);
  UniTensor Spf2cf = Spf.astype(Type.ComplexFloat);
  UniTensor Spf2cd = Spf.astype(Type.ComplexDouble);
  EXPECT_EQ(Spf.dtype(), Type.Float);
  EXPECT_EQ(Spf2d.dtype(), Type.Double);
  EXPECT_EQ(Spf2cf.dtype(), Type.ComplexFloat);
  EXPECT_EQ(Spf2cd.dtype(), Type.ComplexDouble);
}

TEST_F(BlockUniTensorTest, permute1) {
  // rank-3 tensor
  std::vector<cytnx_int64> a = {1, 2, 0};
  auto permuted = UT_permute_1.permute(a, -1);
  for (cytnx_uint64 i = 0; i < 10; i++)
    for (cytnx_uint64 j = 0; j < 6; j++)
      for (cytnx_uint64 k = 0; k < 10; k++) {
        EXPECT_EQ(permuted.at({i, j, k}).exists(), UT_permute_ans1.at({i, j, k}).exists());
        if (permuted.at({i, j, k}).exists())
          EXPECT_EQ(double(permuted.at({i, j, k}).real()),
                    double(UT_permute_ans1.at({i, j, k}).real()));
      }
}

TEST_F(BlockUniTensorTest, permute2) {
  std::vector<cytnx_int64> a = {1, 0};
  auto permuted = UT_permute_2.permute(a, -1);

  for (cytnx_uint64 j = 0; j < 10; j++)
    for (cytnx_uint64 k = 0; k < 10; k++) {
      EXPECT_EQ(permuted.at({j, k}).exists(), UT_permute_ans2.at({j, k}).exists());
      if (permuted.at({j, k}).exists())
        EXPECT_EQ(double(permuted.at({j, k}).real()), double(UT_permute_ans2.at({j, k}).real()));
    }
}

TEST_F(BlockUniTensorTest, permute_1) {
  // rank-3 tensor
  std::vector<cytnx_int64> a = {1, 2, 0};
  auto permuted = UT_permute_1.clone();
  permuted.permute_(a, -1);
  for (cytnx_uint64 i = 0; i < 10; i++)
    for (cytnx_uint64 j = 0; j < 6; j++)
      for (cytnx_uint64 k = 0; k < 10; k++) {
        EXPECT_EQ(permuted.at({i, j, k}).exists(), UT_permute_ans1.at({i, j, k}).exists());
        if (permuted.at({i, j, k}).exists())
          EXPECT_EQ(double(permuted.at({i, j, k}).real()),
                    double(UT_permute_ans1.at({i, j, k}).real()));
      }
}

TEST_F(BlockUniTensorTest, permute_2) {
  std::vector<cytnx_int64> a = {1, 0};
  auto permuted = UT_permute_2.clone();
  permuted.permute_(a, -1);
  for (cytnx_uint64 j = 0; j < 10; j++)
    for (cytnx_uint64 k = 0; k < 10; k++) {
      EXPECT_EQ(permuted.at({j, k}).exists(), UT_permute_ans2.at({j, k}).exists());
      if (permuted.at({j, k}).exists())
        EXPECT_EQ(double(permuted.at({j, k}).real()), double(UT_permute_ans2.at({j, k}).real()));
    }
}

/*=====test info=====
describe:regression test for issue #724 on the BlockUniTensor path. Two
         UniTensors sharing the same underlying block Tensors (via
         relabel(), documented to share data with the original) must not
         corrupt each other's metadata when one of them is permuted in
         place with permute_().
====================*/
TEST_F(BlockUniTensorTest, PermuteInPlaceOnSharedBlockDoesNotCorruptOtherHolder) {
  UniTensor uT = UT_permute_2.clone().set_name("uT");
  UniTensor uT2 = uT.relabel({"a", "b"}).set_name("uT2");

  // Precondition: the two UniTensors really do share the same block storage.
  ASSERT_TRUE(uT.same_data(uT2));
  ASSERT_EQ(uT.Nblocks(), uT2.Nblocks());

  const auto orig_shape = uT.shape();
  const auto orig_labels = uT.labels();
  ASSERT_EQ(orig_shape, std::vector<cytnx_uint64>({10, 10}));
  std::vector<std::vector<double>> orig_vals(10, std::vector<double>(10, 0.0));
  std::vector<std::vector<bool>> orig_exists(10, std::vector<bool>(10, false));
  for (cytnx_int64 j = 0; j < 10; j++)
    for (cytnx_int64 k = 0; k < 10; k++) {
      orig_exists[j][k] = uT.at({j, k}).exists();
      if (orig_exists[j][k]) orig_vals[j][k] = double(uT.at({j, k}).real());
    }

  std::vector<cytnx_int64> a = {1, 0};
  uT2.permute_(a, -1);

  // uT2 changed as expected (matches the pre-existing permute_2 reference answer).
  for (cytnx_int64 j = 0; j < 10; j++)
    for (cytnx_int64 k = 0; k < 10; k++) {
      EXPECT_EQ(uT2.at({j, k}).exists(), UT_permute_ans2.at({j, k}).exists());
      if (uT2.at({j, k}).exists())
        EXPECT_EQ(double(uT2.at({j, k}).real()), double(UT_permute_ans2.at({j, k}).real()));
    }

  // uT must be completely unaffected: shape, labels, and data all preserved. Reading uT after
  // uT2's in-place permute must not even throw (a stale block/qnum mapping vs. a physically
  // permuted shared block Tensor can manifest as an out-of-bound access), so guard each read.
  ASSERT_EQ(uT.shape(), orig_shape);
  EXPECT_EQ(uT.labels(), orig_labels);
  for (cytnx_int64 j = 0; j < 10; j++)
    for (cytnx_int64 k = 0; k < 10; k++) {
      try {
        EXPECT_EQ(uT.at({j, k}).exists(), orig_exists[j][k]);
        if (orig_exists[j][k] && uT.at({j, k}).exists())
          EXPECT_EQ(double(uT.at({j, k}).real()), orig_vals[j][k]);
      } catch (const std::exception& e) {
        ADD_FAILURE() << "uT.at({" << j << "," << k
                      << "}) threw after uT2.permute_() (shared-block metadata corrupted): "
                      << e.what();
      }
    }
}

/*=====test info=====
describe:control test - when blocks are NOT shared (independent clone), an
         in-place permute_() on one BlockUniTensor must not affect the
         other. Guards against a fix that over-isolates or breaks normal
         (non-shared) permute_ behavior.
====================*/
TEST_F(BlockUniTensorTest, PermuteInPlaceOnNonSharedBlockLeavesCloneUnaffected) {
  UniTensor uT = UT_permute_2.clone().set_name("uT");
  UniTensor uT_indep = uT.clone().set_name("uT_indep");

  ASSERT_FALSE(uT.same_data(uT_indep));

  const auto orig_shape = uT.shape();
  std::vector<std::vector<bool>> orig_exists(10, std::vector<bool>(10, false));
  std::vector<std::vector<double>> orig_vals(10, std::vector<double>(10, 0.0));
  for (cytnx_int64 j = 0; j < 10; j++)
    for (cytnx_int64 k = 0; k < 10; k++) {
      orig_exists[j][k] = uT.at({j, k}).exists();
      if (orig_exists[j][k]) orig_vals[j][k] = double(uT.at({j, k}).real());
    }

  std::vector<cytnx_int64> a = {1, 0};
  uT_indep.permute_(a, -1);

  // uT_indep must match the known permute_2 reference answer (proves the permute really ran).
  for (cytnx_int64 j = 0; j < 10; j++)
    for (cytnx_int64 k = 0; k < 10; k++) {
      EXPECT_EQ(uT_indep.at({j, k}).exists(), UT_permute_ans2.at({j, k}).exists());
      if (uT_indep.at({j, k}).exists())
        EXPECT_EQ(double(uT_indep.at({j, k}).real()), double(UT_permute_ans2.at({j, k}).real()));
    }

  // uT (the clone source, independent storage) must be completely unaffected.
  EXPECT_EQ(uT.shape(), orig_shape);
  for (cytnx_int64 j = 0; j < 10; j++)
    for (cytnx_int64 k = 0; k < 10; k++) {
      EXPECT_EQ(uT.at({j, k}).exists(), orig_exists[j][k]);
      if (orig_exists[j][k] && uT.at({j, k}).exists())
        EXPECT_EQ(double(uT.at({j, k}).real()), orig_vals[j][k]);
    }
}

TEST_F(BlockUniTensorTest, contiguous) {
  auto bks = UT_pB_ans.permute({1, 2, 0}).contiguous().get_blocks();

  for (int b = 0; b < bks.size(); b++) {
    int ptr = 0;
    EXPECT_EQ(bks[b].is_contiguous(), true);
    for (cytnx_uint64 i = 0; i < bks[b].shape()[0]; i++)
      for (cytnx_uint64 j = 0; j < bks[b].shape()[1]; j++)
        for (cytnx_uint64 k = 0; k < bks[b].shape()[2]; k++) {
          EXPECT_EQ(double(bks[b].at({i, j, k}).real()), bks[b].storage().at<double>(ptr++));
        }
  }
}

TEST_F(BlockUniTensorTest, contiguous_) {
  auto tmp = UT_pB_ans.permute({1, 2, 0});
  tmp.contiguous_();
  auto bks = tmp.get_blocks();

  for (int b = 0; b < bks.size(); b++) {
    int ptr = 0;
    EXPECT_EQ(bks[b].is_contiguous(), true);
    for (cytnx_uint64 i = 0; i < bks[b].shape()[0]; i++)
      for (cytnx_uint64 j = 0; j < bks[b].shape()[1]; j++)
        for (cytnx_uint64 k = 0; k < bks[b].shape()[2]; k++) {
          EXPECT_EQ(double(bks[b].at({i, j, k}).real()), bks[b].storage().at<double>(ptr++));
        }
  }
}

TEST_F(BlockUniTensorTest, group_basis) {
  auto out = BUT6.group_basis();
  EXPECT_DOUBLE_EQ(double(out.at({0, 0}).real()), double(2));

  EXPECT_DOUBLE_EQ(double(out.at({0, 1}).real()), double(4));

  EXPECT_DOUBLE_EQ(double(out.at({1, 0}).real()), double(3));

  EXPECT_DOUBLE_EQ(double(out.at({1, 1}).real()), double(5));

  EXPECT_DOUBLE_EQ(double(out.at({2, 2}).real()), double(1));

  EXPECT_EQ(out.shape(), std::vector<cytnx_uint64>({3, 3}));
  EXPECT_EQ(out.device(), Device.cpu);
  EXPECT_EQ(out.bonds()[0].qnums(), std::vector<std::vector<cytnx_int64>>({{0}, {1}}));
  EXPECT_EQ(out.bonds()[1].qnums(), std::vector<std::vector<cytnx_int64>>({{0}, {1}}));
}

TEST_F(BlockUniTensorTest, at_for_sparse) {
  BUT6 = BUT6.astype(Type.ComplexDouble);
  auto out = BUT6.at({0, 0});
  EXPECT_DOUBLE_EQ(double(out.real()), double(1));
  EXPECT_DOUBLE_EQ(double(out.imag()), double(0));

  out = BUT6.at({1, 1});
  EXPECT_DOUBLE_EQ(double(out.real()), double(2));
  EXPECT_DOUBLE_EQ(double(out.imag()), double(0));

  out = BUT6.at({1, 2});
  EXPECT_DOUBLE_EQ(double(out.real()), double(4));
  EXPECT_DOUBLE_EQ(double(out.imag()), double(0));

  out = BUT6.at({2, 1});
  EXPECT_DOUBLE_EQ(double(out.real()), double(3));
  EXPECT_DOUBLE_EQ(double(out.imag()), double(0));

  out = BUT6.at({2, 2});
  EXPECT_DOUBLE_EQ(double(out.real()), double(5));
  EXPECT_DOUBLE_EQ(double(out.imag()), double(0));
}

TEST_F(BlockUniTensorTest, get_block_byidx) {
  EXPECT_EQ(AreNearlyEqTensor(UT_pB_ans.get_block(0), t0), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB_ans.get_block(1), t1a), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB_ans.get_block(2), t1b), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB_ans.get_block(3), t2), true);
  // EXPECT_ANY_THROW(UT_pB_ans.get_block({0,0,3}));
}

TEST_F(BlockUniTensorTest, get_block_byqnum) {
  EXPECT_EQ(AreNearlyEqTensor(UT_pB_ans.get_block({0, 0, 0}), t0), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB_ans.get_block({0, 1, 1}), t1a), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB_ans.get_block({1, 0, 1}), t1b), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB_ans.get_block({1, 1, 2}), t2), true);
  // EXPECT_ANY_THROW(UT_pB_ans.get_block({0,0,3}));
}

TEST_F(BlockUniTensorTest, get_blocks) {
  auto bks = UT_pB_ans.get_blocks();
  EXPECT_EQ(AreNearlyEqTensor(bks[0], t0), true);
  EXPECT_EQ(AreNearlyEqTensor(bks[1], t1a), true);
  EXPECT_EQ(AreNearlyEqTensor(bks[2], t1b), true);
  EXPECT_EQ(AreNearlyEqTensor(bks[3], t2), true);
  // EXPECT_ANY_THROW(UT_pB_ans.get_block({0,0,3}));
}

TEST_F(BlockUniTensorTest, get_blocks_) {
  auto bks = UT_pB_ans.get_blocks_();
  EXPECT_EQ(AreNearlyEqTensor(bks[0], t0), true);
  EXPECT_EQ(AreNearlyEqTensor(bks[1], t1a), true);
  EXPECT_EQ(AreNearlyEqTensor(bks[2], t1b), true);
  EXPECT_EQ(AreNearlyEqTensor(bks[3], t2), true);
  EXPECT_EQ(UT_pB_ans.get_block_(0).same_data(bks[0]), true);
  EXPECT_EQ(UT_pB_ans.get_block_(1).same_data(bks[1]), true);
  EXPECT_EQ(UT_pB_ans.get_block_(2).same_data(bks[2]), true);
  EXPECT_EQ(UT_pB_ans.get_block_(3).same_data(bks[3]), true);
  // EXPECT_ANY_THROW(UT_pB_ans.get_block({0,0,3}));
}

TEST_F(BlockUniTensorTest, put_block_byidx) {
  UT_pB.put_block(t0, 0);
  UT_pB.put_block(t1a, 1);
  UT_pB.put_block(t1b, 2);
  UT_pB.put_block(t2, 3);
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 9; j++)
      for (cytnx_uint64 k = 1; k < 30; k++) {
        EXPECT_EQ(UT_pB.at({i, j, k}).exists(), UT_pB_ans.at({i, j, k}).exists());
        if (UT_pB.at({i, j, k}).exists()) EXPECT_EQ(UT_pB.at({i, j, k}), UT_pB_ans.at({i, j, k}));
      }
  EXPECT_EQ(AreNearlyEqTensor(UT_pB.get_block(0), t0), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB.get_block(1), t1a), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB.get_block(2), t1b), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB.get_block(3), t2), true);
  EXPECT_EQ(UT_pB.get_block_(0).same_data(t0), false);
  EXPECT_EQ(UT_pB.get_block_(1).same_data(t1a), false);
  EXPECT_EQ(UT_pB.get_block_(2).same_data(t1b), false);
  EXPECT_EQ(UT_pB.get_block_(3).same_data(t2), false);
  EXPECT_EQ(UT_pB.get_block(0).same_data(t0), false);
  EXPECT_EQ(UT_pB.get_block(1).same_data(t1a), false);
  EXPECT_EQ(UT_pB.get_block(2).same_data(t1b), false);
  EXPECT_EQ(UT_pB.get_block(3).same_data(t2), false);
}

TEST_F(BlockUniTensorTest, put_block__byidx) {
  UT_pB.put_block_(t0, 0);
  UT_pB.put_block_(t1a, 1);
  UT_pB.put_block_(t1b, 2);
  UT_pB.put_block_(t2, 3);
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 9; j++)
      for (cytnx_uint64 k = 1; k < 30; k++) {
        EXPECT_EQ(UT_pB.at({i, j, k}).exists(), UT_pB_ans.at({i, j, k}).exists());
        if (UT_pB.at({i, j, k}).exists()) EXPECT_EQ(UT_pB.at({i, j, k}), UT_pB_ans.at({i, j, k}));
      }
  EXPECT_EQ(AreNearlyEqTensor(UT_pB.get_block(0), t0), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB.get_block(1), t1a), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB.get_block(2), t1b), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB.get_block(3), t2), true);
  EXPECT_EQ(UT_pB.get_block_(0).same_data(t0), true);
  EXPECT_EQ(UT_pB.get_block_(1).same_data(t1a), true);
  EXPECT_EQ(UT_pB.get_block_(2).same_data(t1b), true);
  EXPECT_EQ(UT_pB.get_block_(3).same_data(t2), true);
  EXPECT_EQ(UT_pB.get_block(0).same_data(t0), false);
  EXPECT_EQ(UT_pB.get_block(1).same_data(t1a), false);
  EXPECT_EQ(UT_pB.get_block(2).same_data(t1b), false);
  EXPECT_EQ(UT_pB.get_block(3).same_data(t2), false);
}

TEST_F(BlockUniTensorTest, put_block_byqnum) {
  UT_pB.put_block(t0, {0, 0, 0});
  UT_pB.put_block(t1a, {0, 1, 1});
  UT_pB.put_block(t1b, {1, 0, 1});
  UT_pB.put_block(t2, {1, 1, 2});
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 9; j++)
      for (cytnx_uint64 k = 1; k < 30; k++) {
        EXPECT_EQ(UT_pB.at({i, j, k}).exists(), UT_pB_ans.at({i, j, k}).exists());
        if (UT_pB.at({i, j, k}).exists()) EXPECT_EQ(UT_pB.at({i, j, k}), UT_pB_ans.at({i, j, k}));
      }
  EXPECT_EQ(AreNearlyEqTensor(UT_pB.get_block(0), t0), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB.get_block(1), t1a), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB.get_block(2), t1b), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB.get_block(3), t2), true);
  EXPECT_EQ(UT_pB.get_block_(0).same_data(t0), false);
  EXPECT_EQ(UT_pB.get_block_(1).same_data(t1a), false);
  EXPECT_EQ(UT_pB.get_block_(2).same_data(t1b), false);
  EXPECT_EQ(UT_pB.get_block_(3).same_data(t2), false);
  EXPECT_EQ(UT_pB.get_block(0).same_data(t0), false);
  EXPECT_EQ(UT_pB.get_block(1).same_data(t1a), false);
  EXPECT_EQ(UT_pB.get_block(2).same_data(t1b), false);
  EXPECT_EQ(UT_pB.get_block(3).same_data(t2), false);
}

TEST_F(BlockUniTensorTest, put_block__byqnum) {
  UT_pB.put_block_(t0, {0, 0, 0});
  UT_pB.put_block_(t1a, {0, 1, 1});
  UT_pB.put_block_(t1b, {1, 0, 1});
  UT_pB.put_block_(t2, {1, 1, 2});
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 9; j++)
      for (cytnx_uint64 k = 1; k < 30; k++) {
        EXPECT_EQ(UT_pB.at({i, j, k}).exists(), UT_pB_ans.at({i, j, k}).exists());
        if (UT_pB.at({i, j, k}).exists()) EXPECT_EQ(UT_pB.at({i, j, k}), UT_pB_ans.at({i, j, k}));
      }
  EXPECT_EQ(AreNearlyEqTensor(UT_pB.get_block(0), t0), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB.get_block(1), t1a), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB.get_block(2), t1b), true);
  EXPECT_EQ(AreNearlyEqTensor(UT_pB.get_block(3), t2), true);
  EXPECT_EQ(UT_pB.get_block_(0).same_data(t0), true);
  EXPECT_EQ(UT_pB.get_block_(1).same_data(t1a), true);
  EXPECT_EQ(UT_pB.get_block_(2).same_data(t1b), true);
  EXPECT_EQ(UT_pB.get_block_(3).same_data(t2), true);
  EXPECT_EQ(UT_pB.get_block(0).same_data(t0), false);
  EXPECT_EQ(UT_pB.get_block(1).same_data(t1a), false);
  EXPECT_EQ(UT_pB.get_block(2).same_data(t1b), false);
  EXPECT_EQ(UT_pB.get_block(3).same_data(t2), false);
}

TEST_F(BlockUniTensorTest, reshape) { EXPECT_ANY_THROW(Spf.reshape({4, 1}, 1)); }

TEST_F(BlockUniTensorTest, reshape_) { EXPECT_ANY_THROW(Spf.reshape_({4, 1}, 1)); }

TEST_F(BlockUniTensorTest, contract1) {
  // two sparse matrix

  UT_contract_L1.set_labels({"a", "b"});
  UT_contract_R1.set_labels({"b", "c"});
  UniTensor out = UT_contract_L1.contract(UT_contract_R1);
  auto outbks = out.get_blocks();
  auto ansbks = UT_contract_ans1.get_blocks();
  for (int i = 0; i < ansbks.size(); i++) {
    EXPECT_EQ(AreNearlyEqTensor(outbks[i], ansbks[i], 1e-5), true);
  }
}

TEST_F(BlockUniTensorTest, contract2) {
  // two sparse matrix with degeneracy

  UT_contract_L2.set_labels({"a", "b"});
  UT_contract_R2.set_labels({"b", "c"});
  UniTensor out = UT_contract_L2.contract(UT_contract_R2);
  auto outbks = out.get_blocks();
  auto ansbks = UT_contract_ans2.get_blocks();
  for (int i = 0; i < ansbks.size(); i++) {
    EXPECT_EQ(AreNearlyEqTensor(outbks[i], ansbks[i], 1e-5), true);
  }
}

TEST_F(BlockUniTensorTest, contract3) {
  //// two 3 legs tensor

  UT_contract_L3.set_labels({"a", "b", "c"});
  UT_contract_R3.set_labels({"c", "d", "e"});
  UniTensor out = UT_contract_L3.contract(UT_contract_R3);
  auto outbks = out.get_blocks();
  auto ansbks = UT_contract_ans3.get_blocks();
  for (int i = 0; i < ansbks.size(); i++) {
    EXPECT_EQ(AreNearlyEqTensor(outbks[i], ansbks[i], 1e-5), true);
  }
}

TEST_F(BlockUniTensorTest, contract_mixed_dtype_order_independent) {
  // Reproduce issue #758: real(QN) x complex(QN) should not depend on argument order.
  UniTensor left_real = UT_contract_L2.astype(Type.Double);
  UniTensor right_complex = UT_contract_R2.astype(Type.ComplexDouble);

  left_real.set_labels({"a", "b"});
  right_complex.set_labels({"b", "c"});

  UniTensor out_real_complex;
  UniTensor out_complex_real;
  EXPECT_NO_THROW(out_real_complex = left_real.contract(right_complex));
  EXPECT_NO_THROW(out_complex_real = right_complex.contract(left_real));

  EXPECT_EQ(out_real_complex.dtype(), Type.ComplexDouble);
  EXPECT_EQ(out_complex_real.dtype(), Type.ComplexDouble);

  // Cross-check against all-complex references for each contraction ordering.
  UniTensor left_complex = left_real.astype(Type.ComplexDouble);
  UniTensor right_complex_ref = right_complex.astype(Type.ComplexDouble);
  UniTensor ref_real_complex = left_complex.contract(right_complex_ref);
  UniTensor ref_complex_real = right_complex_ref.contract(left_complex);
  EXPECT_TRUE(AreNearlyEqUniTensor(out_real_complex, ref_real_complex, 1e-10));
  EXPECT_TRUE(AreNearlyEqUniTensor(out_complex_real, ref_complex_real, 1e-10));
}

TEST_F(BlockUniTensorTest, same_data) {
  UniTensor B = UT_pB_ans.permute({1, 0, 2});
  UniTensor C = B.contiguous();
  EXPECT_EQ(B.same_data(C), false);
  EXPECT_EQ(UT_pB_ans.same_data(B), true);
}

TEST_F(BlockUniTensorTest, Add) {
  using namespace std::complex_literals;
  // auto out = BUT4.Add(9+9i);
  // for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
  //   for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
  //     if(out.at({i-1,j-1,k-1,l-1}).exists()){
  //       EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).real()),
  //       double(out.at({i-1,j-1,k-1,l-1}).real())+9);
  //       EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).imag()),
  //       double(out.at({i-1,j-1,k-1,l-1}).imag())+9);
  //     }
  // auto tmp = BUT4;
  // BUT4.Add_(9+9i);
  // for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
  //   for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
  //     if(BUT4.at({i-1,j-1,k-1,l-1}).exists()){
  //       EXPECT_DOUBLE_EQ(double(BUT4.at({i-1,j-1,k-1,l-1}).real()),
  //       double(tmp.at({i-1,j-1,k-1,l-1}).real())+9);
  //       EXPECT_DOUBLE_EQ(double(BUT4.at({i-1,j-1,k-1,l-1}).imag()),
  //       double(tmp.at({i-1,j-1,k-1,l-1}).imag())+9);
  //     }
  BUT4 = BUT4.Load(data_dir + "OriginalBUT.cytnx");
  auto out2 = BUT4.Add(BUT4_2);
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 11; j++)
      for (cytnx_uint64 k = 0; k < 3; k++)
        for (cytnx_uint64 l = 0; l < 5; l++)
          if (out2.at({i, j, k, l}).exists()) {
            EXPECT_DOUBLE_EQ(double(out2.at({i, j, k, l}).real()),
                             double(BUTpT2.at({i, j, k, l}).real()));
            EXPECT_DOUBLE_EQ(double(out2.at({i, j, k, l}).imag()),
                             double(BUTpT2.at({i, j, k, l}).imag()));
          }
  BUT4.Add_(BUT4_2);
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 11; j++)
      for (cytnx_uint64 k = 0; k < 3; k++)
        for (cytnx_uint64 l = 0; l < 5; l++)
          if (BUT4.at({i, j, k, l}).exists()) {
            EXPECT_DOUBLE_EQ(double(BUT4.at({i, j, k, l}).real()),
                             double(BUTpT2.at({i, j, k, l}).real()));
            EXPECT_DOUBLE_EQ(double(BUT4.at({i, j, k, l}).imag()),
                             double(BUTpT2.at({i, j, k, l}).imag()));
          }
}

TEST_F(BlockUniTensorTest, Mul) {
  auto out = BUT4.Mul(9);
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 11; j++)
      for (cytnx_uint64 k = 0; k < 3; k++)
        for (cytnx_uint64 l = 0; l < 5; l++)
          if (out.at({i, j, k, l}).exists()) {
            EXPECT_DOUBLE_EQ(double(out.at({i, j, k, l}).real()),
                             double(BUTm9.at({i, j, k, l}).real()));
            EXPECT_DOUBLE_EQ(double(out.at({i, j, k, l}).imag()),
                             double(BUTm9.at({i, j, k, l}).imag()));
          }
  BUT4.Mul_(9);
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 11; j++)
      for (cytnx_uint64 k = 0; k < 3; k++)
        for (cytnx_uint64 l = 0; l < 5; l++)
          if (BUT4.at({i, j, k, l}).exists()) {
            EXPECT_DOUBLE_EQ(double(BUT4.at({i, j, k, l}).real()),
                             double(BUTm9.at({i, j, k, l}).real()));
            EXPECT_DOUBLE_EQ(double(BUT4.at({i, j, k, l}).imag()),
                             double(BUTm9.at({i, j, k, l}).imag()));
          }
}

TEST_F(BlockUniTensorTest, Sub) {
  using namespace std::complex_literals;
  // auto out = BUT4.Sub(9+9i);
  // for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
  //   for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
  //     if(out.at({i-1,j-1,k-1,l-1}).exists()){
  //       EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).real()),
  //       double(out.at({i-1,j-1,k-1,l-1}).real())-9);
  //       EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).imag()),
  //       double(out.at({i-1,j-1,k-1,l-1}).imag())-9);
  //     }
  // auto tmp = BUT4;
  // BUT4.Sub_(9+9i);
  // for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
  //   for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
  //     if(BUT4.at({i-1,j-1,k-1,l-1}).exists()){
  //       EXPECT_DOUBLE_EQ(double(BUT4.at({i-1,j-1,k-1,l-1}).real()),
  //       double(tmp.at({i-1,j-1,k-1,l-1}).real())-9);
  //       EXPECT_DOUBLE_EQ(double(BUT4.at({i-1,j-1,k-1,l-1}).imag()),
  //       double(tmp.at({i-1,j-1,k-1,l-1}).imag())-9);
  //     }
  BUT4 = BUT4.Load(data_dir + "OriginalBUT.cytnx");
  auto out2 = BUT4.Sub(BUT4_2);
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 11; j++)
      for (cytnx_uint64 k = 0; k < 3; k++)
        for (cytnx_uint64 l = 0; l < 5; l++)
          if (out2.at({i, j, k, l}).exists()) {
            EXPECT_DOUBLE_EQ(double(out2.at({i, j, k, l}).real()),
                             double(BUTsT2.at({i, j, k, l}).real()));
            EXPECT_DOUBLE_EQ(double(out2.at({i, j, k, l}).imag()),
                             double(BUTsT2.at({i, j, k, l}).imag()));
          }
  BUT4.Sub_(BUT4_2);
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 11; j++)
      for (cytnx_uint64 k = 0; k < 3; k++)
        for (cytnx_uint64 l = 0; l < 5; l++)
          if (BUT4.at({i, j, k, l}).exists()) {
            EXPECT_DOUBLE_EQ(double(BUT4.at({i, j, k, l}).real()),
                             double(BUTsT2.at({i, j, k, l}).real()));
            EXPECT_DOUBLE_EQ(double(BUT4.at({i, j, k, l}).imag()),
                             double(BUTsT2.at({i, j, k, l}).imag()));
          }
}

TEST_F(BlockUniTensorTest, Div) {
  auto out = BUT4.Div(9);
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 11; j++)
      for (cytnx_uint64 k = 0; k < 3; k++)
        for (cytnx_uint64 l = 0; l < 5; l++)
          if (out.at({i, j, k, l}).exists()) {
            EXPECT_DOUBLE_EQ(double(out.at({i, j, k, l}).real()),
                             double(BUTd9.at({i, j, k, l}).real()));
            EXPECT_DOUBLE_EQ(double(out.at({i, j, k, l}).imag()),
                             double(BUTd9.at({i, j, k, l}).imag()));
          }
  BUT4.Div_(9);
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 11; j++)
      for (cytnx_uint64 k = 0; k < 3; k++)
        for (cytnx_uint64 l = 0; l < 5; l++)
          if (BUT4.at({i, j, k, l}).exists()) {
            EXPECT_DOUBLE_EQ(double(BUT4.at({i, j, k, l}).real()),
                             double(BUTd9.at({i, j, k, l}).real()));
            EXPECT_DOUBLE_EQ(double(BUT4.at({i, j, k, l}).imag()),
                             double(BUTd9.at({i, j, k, l}).imag()));
          }

  // BUT4 = BUT4.Load("OriginalBUT.cytnx");
  // auto out2 = BUT4.Div(BUT4_2);
  // for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
  //   for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
  //     if(out2.at({i-1,j-1,k-1,l-1}).exists()){
  //       EXPECT_DOUBLE_EQ(double(out2.at({i-1,j-1,k-1,l-1}).real()),
  //       double(BUTdT2.at({i-1,j-1,k-1,l-1}).real()));
  //       EXPECT_DOUBLE_EQ(double(out2.at({i-1,j-1,k-1,l-1}).imag()),
  //       double(BUTdT2.at({i-1,j-1,k-1,l-1}).imag()));
  //     }
  // BUT4.Div_(BUT4_2);
  // for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
  //   for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
  //     if(BUT4.at({i-1,j-1,k-1,l-1}).exists()){
  //       EXPECT_DOUBLE_EQ(double(BUT4.at({i-1,j-1,k-1,l-1}).real()),
  //       double(BUTdT2.at({i-1,j-1,k-1,l-1}).real()));
  //       EXPECT_DOUBLE_EQ(double(BUT4.at({i-1,j-1,k-1,l-1}).imag()),
  //       double(BUTdT2.at({i-1,j-1,k-1,l-1}).imag()));
  //     }
}

TEST_F(BlockUniTensorTest, LinAlgElementwise) {
  const double tol = 1e-14;
  UniTensor T = BUT4;
  EXPECT_EQ(AreNearlyEqUniTensor(2. * BUT4, T + T, tol), true);
  EXPECT_EQ(AreNearlyEqUniTensor(2. * BUT4, T + BUT4, tol), true);
  EXPECT_EQ(AreNearlyEqUniTensor((T + T + T + T) / 4., BUT4, tol), true);
  EXPECT_EQ(AreNearlyEqUniTensor((T + T + BUT4 + T) / 4., BUT4, tol), true);
  EXPECT_EQ(AreNearlyEqUniTensor((2 * T) - T, BUT4, tol), true);
  EXPECT_EQ(AreNearlyEqUniTensor((2 * T) - BUT4, BUT4, tol), true);
  EXPECT_EQ(AreNearlyEqUniTensor((T * T * T) / T, BUT4 * BUT4, tol), true);
  // test inline
  UniTensor T2 = T.clone();
  T2 += T;
  T2 /= 2.;
  EXPECT_EQ(AreNearlyEqUniTensor(T2, T, tol), true);
  // test Mul and Div for tensors
  UniTensor Tsq = T.clone();
  Tsq *= T;
  EXPECT_EQ(AreNearlyEqUniTensor(Tsq, T * T, tol), true);
  EXPECT_EQ(AreNearlyEqUniTensor(Tsq, T.Pow(2.), tol), true);
  Tsq /= T;
  EXPECT_EQ(AreNearlyEqUniTensor(Tsq, T, tol), true);
}

TEST_F(BlockUniTensorTest, Norm) {
  // EXPECT_TRUE(Scalar(BUT4.Norm().item()-10.02330912178208).abs()<1e-5);
  Tensor but_norm = BUT4.Norm();
  EXPECT_TRUE(but_norm.is_scalar());
  EXPECT_DOUBLE_EQ(double(but_norm.item().real()), 10.36019459497064);

  Tensor diag_norm = UT_diag.Norm();
  EXPECT_TRUE(diag_norm.is_scalar());
  cytnx_double tmp = double(diag_norm.item().real());
  cytnx_double ans = 0;
  for (cytnx_uint64 i = 0; i < UT_diag.bonds()[0].qnums().size(); i++) {
    cytnx_uint64 deg = UT_diag.bonds()[0]._impl->_degs[i];
    for (int j = 0; j < deg; j++) ans += (i + 1) * (i + 1);
  }
  ans = sqrt(ans);
  EXPECT_DOUBLE_EQ(ans, tmp);
}

/*=====test info=====
describe:test pseudo-inverse
====================*/
TEST_F(BlockUniTensorTest, Inv) {
  const double tol = 1e-14;
  double clip = 1e-14;
  EXPECT_TRUE(AreNearlyEqUniTensor(BUT4.Inv(clip).Inv_(clip), BUT4, tol));
  EXPECT_FALSE(AreNearlyEqUniTensor(BUT4.Inv(clip), BUT4, tol));
  clip = 0.1;  // test actual clipping as well
  auto tmp = BUT4.clone();
  tmp.Inv_(clip);  // test inline version
  EXPECT_TRUE(AreEqUniTensor(BUT4.Inv(clip), tmp));
  tmp = BUT4.clone();
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 11; j++)
      for (cytnx_uint64 k = 0; k < 3; k++)
        for (cytnx_uint64 l = 0; l < 5; l++) {
          auto proxy = tmp.at({i, j, k, l});
          if (proxy.exists()) {
            Scalar val = proxy;
            if (val.abs() <= clip)
              proxy = cytnx_complex128(0., 0.);
            else
              proxy = cytnx_complex128(1., 0.) / proxy;
          }
        }
  EXPECT_TRUE(AreNearlyEqUniTensor(BUT4.Inv(clip), tmp, tol));
}

/*=====test info=====
describe:test power
====================*/
TEST_F(BlockUniTensorTest, Pow) {
  const double tol = 1e-14;
  EXPECT_TRUE(AreNearlyEqUniTensor(BUT4.Pow(2.), BUT4 * BUT4, tol));
  auto tmp = BUT4.clone();
  tmp.Pow_(2.3);  // test inline version
  EXPECT_TRUE(AreEqUniTensor(BUT4.Pow(2.3), tmp));
  for (double p = 0.; p < 1.6; p += 0.5) {
    tmp = BUT4.clone();
    for (cytnx_uint64 i = 0; i < 5; i++)
      for (cytnx_uint64 j = 0; j < 11; j++)
        for (cytnx_uint64 k = 0; k < 3; k++)
          for (cytnx_uint64 l = 0; l < 5; l++) {
            auto proxy = tmp.at({i, j, k, l});
            if (proxy.exists()) {
              Scalar val = proxy;
              proxy =
                std::pow(cytnx_complex128((cytnx_double)val.real(), (cytnx_double)val.imag()), p);
            }
          }
    EXPECT_TRUE(AreNearlyEqUniTensor(BUT4.Pow(p), tmp, tol));
  }
}

TEST_F(BlockUniTensorTest, Conj) {
  auto tmp = BUT4.Conj();
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 11; j++)
      for (cytnx_uint64 k = 0; k < 3; k++)
        for (cytnx_uint64 l = 0; l < 5; l++)
          if (BUT4.at({i, j, k, l}).exists()) {
            // EXPECT_TRUE(Scalar(tmp.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
            EXPECT_DOUBLE_EQ(double(tmp.at({i, j, k, l}).real()),
                             double(BUT4.at({i, j, k, l}).real()));
            EXPECT_DOUBLE_EQ(double(tmp.at({i, j, k, l}).imag()),
                             -double(BUT4.at({i, j, k, l}).imag()));
          }
  tmp = BUT4.clone();
  tmp.Conj_();
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 11; j++)
      for (cytnx_uint64 k = 0; k < 3; k++)
        for (cytnx_uint64 l = 0; l < 5; l++)
          if (BUT4.at({i, j, k, l}).exists()) {
            // EXPECT_TRUE(Scalar(BUT4.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
            EXPECT_DOUBLE_EQ(double(BUT4.at({i, j, k, l}).real()),
                             double(tmp.at({i, j, k, l}).real()));
            EXPECT_DOUBLE_EQ(double(BUT4.at({i, j, k, l}).imag()),
                             -double(tmp.at({i, j, k, l}).imag()));
          }

  tmp = UT_diag_cplx.Conj();
  for (cytnx_uint64 i = 0; i < UT_diag.bonds()[0].qnums().size(); i++) {
    cytnx_uint64 deg = UT_diag.bonds()[0]._impl->_degs[i];
    for (cytnx_uint64 j = 0; j < deg; j++) {
      EXPECT_DOUBLE_EQ(double(tmp.get_block_(i).at({j}).real()),
                       double(UT_diag_cplx.get_block_(i).at({j}).real()));
      EXPECT_DOUBLE_EQ(double(tmp.get_block_(i).at({j}).imag()),
                       -double(UT_diag_cplx.get_block_(i).at({j}).imag()));
    }
  }
}

TEST_F(BlockUniTensorTest, Transpose) {
  auto tmp = BUT1.Transpose().set_name("BUT1.Transpose");
  EXPECT_EQ(tmp.bonds()[0].type(), BD_IN);
  EXPECT_EQ(tmp.bonds()[1].type(), BD_IN);
  EXPECT_EQ(tmp.bonds()[2].type(), BD_OUT);
  EXPECT_EQ(tmp.bonds()[3].type(), BD_OUT);

  tmp = BUT5.Transpose();
  EXPECT_EQ(tmp.bonds()[0].type(), BD_KET);
  EXPECT_EQ(tmp.bonds()[1].type(), BD_BRA);
  EXPECT_EQ(tmp.bonds()[0].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));
  EXPECT_EQ(tmp.bonds()[1].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));

  BUT1.Transpose_();
  EXPECT_EQ(BUT1.bonds()[0].type(), BD_IN);
  EXPECT_EQ(BUT1.bonds()[1].type(), BD_IN);
  EXPECT_EQ(BUT1.bonds()[2].type(), BD_OUT);
  EXPECT_EQ(BUT1.bonds()[3].type(), BD_OUT);

  BUT5.Transpose_();
  EXPECT_EQ(BUT5.bonds()[0].type(), BD_KET);
  EXPECT_EQ(BUT5.bonds()[1].type(), BD_BRA);
  EXPECT_EQ(BUT5.bonds()[0].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));
  EXPECT_EQ(BUT5.bonds()[1].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));
}

TEST_F(BlockUniTensorTest, Trace) {
  auto tmp = BUT4.Trace(0, 3);
  for (cytnx_uint64 j = 0; j < 11; j++)
    for (cytnx_uint64 k = 0; k < 3; k++)
      if (BUtrT4.at({j, k}).exists()) {
        // EXPECT_TRUE(Scalar(tmp.at({j-1,k-1})-BUtrT4.at({j-1,k-1})).abs()<1e-5);
        EXPECT_DOUBLE_EQ(double(tmp.at({j, k}).real()), double(BUtrT4.at({j, k}).real()));
        EXPECT_DOUBLE_EQ(double(tmp.at({j, k}).imag()), double(BUtrT4.at({j, k}).imag()));
      }
  tmp = UT_diag.Trace(0, 1);
  cytnx_double ans = 0;
  for (cytnx_uint64 i = 0; i < UT_diag.bonds()[0].qnums().size(); i++) {
    cytnx_uint64 deg = UT_diag.bonds()[0]._impl->_degs[i];
    for (int j = 0; j < deg; j++) ans += i + 1;
  }
  EXPECT_EQ(tmp.rank(), 0);
  EXPECT_EQ(tmp.rowrank(), 0);
  EXPECT_TRUE(tmp.bonds().empty());
  EXPECT_TRUE(tmp.shape().empty());
  EXPECT_EQ(tmp.syms(), UT_diag.syms());
  EXPECT_FALSE(tmp.is_diag());
  EXPECT_TRUE(tmp.get_block_().is_scalar());
  EXPECT_TRUE(tmp.get_block_({}).is_scalar());
  EXPECT_DOUBLE_EQ(double(tmp.at({}).real()), double(ans));
  EXPECT_DOUBLE_EQ(double(tmp.at({}).imag()), double(0));
  EXPECT_NO_THROW(tmp.to_dense());
  testing::internal::CaptureStdout();
  EXPECT_NO_THROW(tmp.print_block(0, false));
  EXPECT_NE(testing::internal::GetCapturedStdout().find("rank-0 scalar block"), std::string::npos);

  UniTensor loaded_scalar;
  tmp.Save(temp_file_path);
  EXPECT_NO_THROW(loaded_scalar = UniTensor::Load(temp_file_path));
  EXPECT_EQ(loaded_scalar.uten_type(), UTenType.Block);
  EXPECT_EQ(loaded_scalar.rank(), 0);
  EXPECT_EQ(loaded_scalar.syms(), UT_diag.syms());
  EXPECT_DOUBLE_EQ(double(loaded_scalar.at({}).real()), double(ans));

  UniTensor scalar_contract;
  EXPECT_NO_THROW(scalar_contract = tmp.contract(loaded_scalar));
  EXPECT_EQ(scalar_contract.uten_type(), UTenType.Block);
  EXPECT_EQ(scalar_contract.rank(), 0);
  EXPECT_EQ(scalar_contract.syms(), UT_diag.syms());
  EXPECT_DOUBLE_EQ(double(scalar_contract.at({}).real()), double(ans * ans));

  tmp = UT_diag.clone();
  tmp.Trace_(0, 1);
  EXPECT_EQ(tmp.uten_type(), UTenType.Block);
  EXPECT_EQ(tmp.rank(), 0);
  EXPECT_EQ(tmp.rowrank(), 0);
  EXPECT_TRUE(tmp.bonds().empty());
  EXPECT_TRUE(tmp.shape().empty());
  EXPECT_EQ(tmp.syms(), UT_diag.syms());
  EXPECT_FALSE(tmp.is_diag());
  EXPECT_TRUE(tmp.get_block_().is_scalar());
  EXPECT_DOUBLE_EQ(double(tmp.at({}).real()), double(ans));
  EXPECT_DOUBLE_EQ(double(tmp.at({}).imag()), double(0));
  EXPECT_NO_THROW(tmp.to_dense());

  UniTensor transposed = tmp.Transpose();
  EXPECT_EQ(transposed.uten_type(), UTenType.Block);
  EXPECT_EQ(transposed.rank(), 0);
  EXPECT_EQ(transposed.rowrank(), 0);
  EXPECT_TRUE(transposed.bonds().empty());
  EXPECT_TRUE(transposed.shape().empty());
  EXPECT_EQ(transposed.syms(), UT_diag.syms());
  EXPECT_FALSE(transposed.is_diag());
  EXPECT_TRUE(transposed.get_block_().is_scalar());
  EXPECT_DOUBLE_EQ(double(transposed.at({}).real()), double(ans));
  EXPECT_DOUBLE_EQ(double(transposed.at({}).imag()), double(0));

  tmp.Transpose_();
  EXPECT_EQ(tmp.uten_type(), UTenType.Block);
  EXPECT_EQ(tmp.rank(), 0);
  EXPECT_EQ(tmp.rowrank(), 0);
  EXPECT_TRUE(tmp.bonds().empty());
  EXPECT_TRUE(tmp.shape().empty());
  EXPECT_EQ(tmp.syms(), UT_diag.syms());
  EXPECT_FALSE(tmp.is_diag());
  EXPECT_TRUE(tmp.get_block_().is_scalar());
  EXPECT_DOUBLE_EQ(double(tmp.at({}).real()), double(ans));
  EXPECT_DOUBLE_EQ(double(tmp.at({}).imag()), double(0));

  EXPECT_NO_THROW(BUT1.Trace(0, 3));
  EXPECT_THROW(BUT1.Trace(), std::logic_error);
  EXPECT_THROW(BUT1.Trace(0, 1), std::logic_error);
  EXPECT_THROW(BUT1.Trace(-1, 2), std::logic_error);
  EXPECT_THROW(BUT1.Trace(-1, 5), std::logic_error);

  EXPECT_NO_THROW(BUT1.Trace("0", "3"));
  EXPECT_THROW(BUT1.Trace(), std::logic_error);
  EXPECT_THROW(BUT1.Trace("0", "1"), std::logic_error);
  EXPECT_THROW(BUT1.Trace("-1", "2"), std::logic_error);
  EXPECT_THROW(BUT1.Trace("-1", "5"), std::logic_error);
}

TEST_F(BlockUniTensorTest, Dagger) {
  auto tmp = BUT1.Dagger();
  EXPECT_EQ(tmp.bonds()[0].type(), BD_IN);
  EXPECT_EQ(tmp.bonds()[1].type(), BD_IN);
  EXPECT_EQ(tmp.bonds()[2].type(), BD_OUT);
  EXPECT_EQ(tmp.bonds()[3].type(), BD_OUT);

  tmp = BUT5.Dagger();
  EXPECT_EQ(tmp.bonds()[0].type(), BD_KET);
  EXPECT_EQ(tmp.bonds()[1].type(), BD_BRA);
  EXPECT_EQ(tmp.bonds()[0].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));
  EXPECT_EQ(tmp.bonds()[1].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));

  BUT1.Dagger_();
  EXPECT_EQ(BUT1.bonds()[0].type(), BD_IN);
  EXPECT_EQ(BUT1.bonds()[1].type(), BD_IN);
  EXPECT_EQ(BUT1.bonds()[2].type(), BD_OUT);
  EXPECT_EQ(BUT1.bonds()[3].type(), BD_OUT);

  BUT5.Dagger_();
  EXPECT_EQ(BUT5.bonds()[0].type(), BD_KET);
  EXPECT_EQ(BUT5.bonds()[1].type(), BD_BRA);
  EXPECT_EQ(BUT5.bonds()[0].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));
  EXPECT_EQ(BUT5.bonds()[1].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));

  tmp = BUT4.Dagger().set_name("BUT4.Dagger");
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 11; j++)
      for (cytnx_uint64 k = 0; k < 3; k++)
        for (cytnx_uint64 l = 0; l < 5; l++) {
          if (BUT4.at({i, j, k, l}).exists()) {
            EXPECT_DOUBLE_EQ(double(tmp.at({l, k, j, i}).real()),
                             double(BUT4.at({i, j, k, l}).real()));
            EXPECT_DOUBLE_EQ(double(tmp.at({l, k, j, i}).imag()),
                             -double(BUT4.at({i, j, k, l}).imag()));
          } else {
            EXPECT_FALSE(tmp.at({l, k, j, i}).exists());
          }
        }
  tmp = BUT4.clone();
  tmp.Dagger_();
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 11; j++)
      for (cytnx_uint64 k = 0; k < 3; k++)
        for (cytnx_uint64 l = 0; l < 5; l++) {
          if (BUT4.at({i, j, k, l}).exists()) {
            // EXPECT_TRUE(Scalar(BUT4.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
            EXPECT_DOUBLE_EQ(double(BUT4.at({i, j, k, l}).real()),
                             double(tmp.at({l, k, j, i}).real()));
            EXPECT_DOUBLE_EQ(double(BUT4.at({i, j, k, l}).imag()),
                             -double(tmp.at({l, k, j, i}).imag()));
          } else {
            EXPECT_FALSE(tmp.at({l, k, j, i}).exists());
          }
        }

  tmp = UT_pB.set_rowrank(2).Dagger().set_name("UT_pB.Dagger");
  EXPECT_EQ(tmp.rowrank(), 1);
  EXPECT_EQ(tmp.bonds()[0].type(), BD_IN);
  EXPECT_EQ(tmp.bonds()[1].type(), BD_OUT);
  EXPECT_EQ(tmp.bonds()[2].type(), BD_OUT);
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 9; j++)
      for (cytnx_uint64 k = 0; k < 30; k++) {
        if (UT_pB.at({i, j, k}).exists()) {
          EXPECT_DOUBLE_EQ(double(tmp.at({k, j, i}).real()), double(UT_pB.at({i, j, k}).real()));
        } else {
          EXPECT_FALSE(tmp.at({k, j, i}).exists());
        }
      }

  tmp = UT_diag_cplx.Dagger();
  for (cytnx_uint64 i = 0; i < UT_diag_cplx.bonds()[0].qnums().size(); i++) {
    cytnx_uint64 deg = UT_diag_cplx.bonds()[0]._impl->_degs[i];
    for (cytnx_uint64 j = 0; j < deg; j++) {
      EXPECT_DOUBLE_EQ(double(tmp.get_block_(i).at({j}).real()),
                       double(UT_diag_cplx.get_block_(i).at({j}).real()));
      EXPECT_DOUBLE_EQ(double(tmp.get_block_(i).at({j}).imag()),
                       -double(UT_diag_cplx.get_block_(i).at({j}).imag()));
    }
  }
}

TEST_F(BlockUniTensorTest, elem_exist) {
  for (cytnx_uint64 i = 0; i < 5; i++)
    for (cytnx_uint64 j = 0; j < 11; j++)
      for (cytnx_uint64 k = 0; k < 3; k++)
        for (cytnx_uint64 l = 0; l < 5; l++)
          if (BUT4.elem_exists({i, j, k, l})) {
            cytnx_int64 _a;
            std::vector<cytnx_uint64> _b;
            ((BlockUniTensor*)BUT4._impl.get())->_fx_locate_elem(_a, _b, {i, j, k, l});
            std::vector<cytnx_uint64> qind = BUT4.get_qindices(_a);
            EXPECT_EQ(BUT4.bonds()[0].qnums()[qind[0]][0] - BUT4.bonds()[1].qnums()[qind[1]][0] +
                        BUT4.bonds()[2].qnums()[qind[2]][0] - BUT4.bonds()[3].qnums()[qind[3]][0],
                      0);
          }

  cytnx_uint64 offset = 0;
  for (cytnx_uint64 i = 0; i < UT_diag_cplx.bonds()[0].qnums().size(); i++) {
    cytnx_uint64 deg = UT_diag_cplx.bonds()[0]._impl->_degs[i];
    for (cytnx_uint64 j = 0; j < deg; j++) {
      EXPECT_TRUE(UT_diag_cplx.elem_exists({offset + j, offset + j}));
      EXPECT_DOUBLE_EQ(double(UT_diag_cplx.at({offset + j, offset + j}).real()), double(i + 1));
      EXPECT_DOUBLE_EQ(double(UT_diag_cplx.at({offset + j, offset + j}).imag()), double(i + 1));
    }
    offset += deg;
  }

  EXPECT_THROW(BUT4.elem_exists({100, 0, 0, 0}), std::logic_error);
  EXPECT_THROW(BUT4.elem_exists({1, 0, 0, 0, 0}), std::logic_error);
  EXPECT_THROW(BUT4.elem_exists({0, 0, 0}), std::logic_error);
  EXPECT_THROW(BUT4.elem_exists({}), std::logic_error);
}

TEST_F(BlockUniTensorTest, truncate) {
  auto tmp = BUT5.truncate(0, 1);
  EXPECT_EQ(tmp.bonds()[0].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 6}, {0, 1}}));
  EXPECT_EQ(tmp.bonds()[1].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));
  tmp = BUT5.truncate(1, 0);
  EXPECT_EQ(tmp.bonds()[0].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));
  EXPECT_EQ(tmp.bonds()[1].qnums(),
            std::vector<std::vector<cytnx_int64>>({{1, 5}, {1, 6}, {0, 1}}));
  BUT5.truncate_(1, 3);
  EXPECT_EQ(BUT5.bonds()[0].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));
  EXPECT_EQ(BUT5.bonds()[1].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}}));
  EXPECT_THROW(BUT5.truncate(-1, 1), std::logic_error);
  EXPECT_THROW(BUT5.truncate(0, -1), std::logic_error);
  EXPECT_THROW(BUT5.truncate(0, 4), std::logic_error);
  EXPECT_THROW(BUT5.truncate(2, 0), std::logic_error);
  EXPECT_THROW(BUT5.truncate_(-1, 1), std::logic_error);
  EXPECT_THROW(BUT5.truncate_(0, -1), std::logic_error);
  EXPECT_THROW(BUT5.truncate_(0, 4), std::logic_error);
  EXPECT_THROW(BUT5.truncate_(2, 0), std::logic_error);
}

TEST_F(BlockUniTensorTest, get_qindices) {
  auto out = BUT6.get_qindices(0);
  EXPECT_EQ(out.size(), 2);
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(out[1], 0);

  out = BUT6.get_qindices(1);
  EXPECT_EQ(out.size(), 2);
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], 1);

  out = BUT6.get_qindices(2);
  EXPECT_EQ(out.size(), 2);
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], 2);

  out = BUT6.get_qindices(3);
  EXPECT_EQ(out.size(), 2);
  EXPECT_EQ(out[0], 2);
  EXPECT_EQ(out[1], 1);

  out = BUT6.get_qindices(4);
  EXPECT_EQ(out.size(), 2);
  EXPECT_EQ(out[0], 2);
  EXPECT_EQ(out[1], 2);
}

TEST_F(BlockUniTensorTest, get_itoi) {
  auto out = BUT6.get_itoi();
  EXPECT_EQ(out.size(), 5);
  EXPECT_EQ(out[0], std::vector<cytnx_uint64>({0, 0}));
  EXPECT_EQ(out[1], std::vector<cytnx_uint64>({1, 1}));
  EXPECT_EQ(out[2], std::vector<cytnx_uint64>({1, 2}));
  EXPECT_EQ(out[3], std::vector<cytnx_uint64>({2, 1}));
  EXPECT_EQ(out[4], std::vector<cytnx_uint64>({2, 2}));
}

TEST_F(BlockUniTensorTest, get_bond_ref) {
  /*
  Bond B1g = Bond(BD_IN, {Qs(1), Qs(0), Qs(0)}, {1, 1, 1});
  Bond B2g = Bond(BD_OUT, {Qs(1), Qs(0), Qs(0)}, {1, 1, 1});
  UniTensor BUT6 = UniTensor({B1g, B2g});
  */
  auto bd1 = BUT6.bond_(0);
  auto bd2 = BUT6.bond_("1");
  auto ut1 = UniTensor({bd1, bd2});
  EXPECT_EQ(ut1.bonds().size(), 2);
  EXPECT_EQ(ut1.bonds()[0].qnums(), std::vector<std::vector<cytnx_int64>>({{1}, {0}, {0}}));
  EXPECT_EQ(ut1.bonds()[1].qnums(), std::vector<std::vector<cytnx_int64>>({{1}, {0}, {0}}));
}

// ============ convert_from / from_ ============

// Dense <-> Block round-trip: full block content (incl. within-block off-diagonal) is preserved,
// symmetry-forbidden entries stay zero, and convert_from returns *this for chaining.
TEST_F(BlockUniTensorTest, convert_from_dense_block_roundtrip) {
  Bond bi = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 2});
  UniTensor B = UniTensor({bi, bi.redirect()});
  B.at({0, 0}) = 1.0;
  B.at({0, 1}) = 5.0;  // within the charge-0 block (off-diagonal, allowed)
  B.at({1, 1}) = 2.0;
  B.at({2, 2}) = 3.0;  // within the charge-1 block
  B.at({3, 3}) = 4.0;

  UniTensor D = UniTensor(zeros(B.shape()));
  UniTensor& ret = D.convert_from(B);
  EXPECT_EQ(&ret, &D);  // returns reference to *this
  EXPECT_DOUBLE_EQ(double(D.at({0, 0}).real()), 1.0);
  EXPECT_DOUBLE_EQ(double(D.at({0, 1}).real()), 5.0);
  EXPECT_DOUBLE_EQ(double(D.at({2, 2}).real()), 3.0);
  EXPECT_DOUBLE_EQ(double(D.at({0, 2}).real()), 0.0);  // forbidden sector stays zero

  UniTensor B2 = UniTensor({bi, bi.redirect()});
  B2.convert_from(D);
  EXPECT_TRUE(AreEqUniTensor(B, B2));  // round-trip recovers the original block exactly
}

// Dense -> Block honors tol in all cases: a nonzero symmetry-forbidden entry is rejected at the
// default tol=0, but a large tol or force=true tolerates it (the forbidden entry is dropped and the
// allowed entries reproduce the original block exactly).
TEST_F(BlockUniTensorTest, convert_from_tol_forbidden_nonzero) {
  Bond bi = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 2});
  UniTensor B = UniTensor({bi, bi.redirect()});
  B.at({0, 0}) = 1.0;
  B.at({0, 1}) = 5.0;
  B.at({1, 1}) = 2.0;
  B.at({2, 2}) = 3.0;
  B.at({3, 3}) = 4.0;

  // Dense form of B with one extra nonzero in a symmetry-forbidden position.
  UniTensor D = UniTensor(zeros(B.shape()));
  D.convert_from(B);
  D.at({0, 2}) = 7.0;  // forbidden sector (charge-0 row, charge-1 col)

  UniTensor B0 = UniTensor({bi, bi.redirect()});
  EXPECT_ANY_THROW(B0.convert_from(D));  // tol defaults to 0 -> rejected

  UniTensor Bf = UniTensor({bi, bi.redirect()});
  Bf.convert_from(D, true);  // force drops the forbidden entry
  EXPECT_TRUE(AreEqUniTensor(Bf, B));

  UniTensor Bt = UniTensor({bi, bi.redirect()});
  Bt.convert_from(D, false, 10.0);  // large tol tolerates it, same result
  EXPECT_TRUE(AreEqUniTensor(Bt, B));
}

// Converting a Block into a diagonal Dense is not supported and must throw.
TEST_F(BlockUniTensorTest, convert_from_diagonal_dense_target_throws) {
  Bond bi = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1});
  UniTensor B = UniTensor({bi, bi.redirect()});
  B.at({0, 0}) = 1.0;
  B.at({1, 1}) = 2.0;

  UniTensor Ddiag = UniTensor(zeros(2), true);  // diagonal Dense, shape (2,2)
  EXPECT_ANY_THROW(Ddiag.convert_from(B));
}

// ============ to_dense / to_dense_ ============

// A diagonal BlockUniTensor is expanded to a full one: each rank-1 (diagonal) block becomes a
// diagonal matrix and is_diag() becomes false; the in-place variant gives the same result.
TEST_F(BlockUniTensorTest, to_dense_diag) {
  Bond bd = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 3});
  for (auto dtype : {Type.ComplexDouble, Type.Double}) {
    UniTensor B = UniTensor({bd, bd.redirect()}, {"a", "b"}, 1, dtype, Device.cpu, true);
    random::uniform_(B, -10.0, 10.0, 0);

    UniTensor dense = B.to_dense();
    EXPECT_TRUE(B.is_diag());
    EXPECT_FALSE(dense.is_diag());
    EXPECT_EQ(dense.dtype(), dtype);
    ASSERT_EQ(dense.Nblocks(), B.Nblocks());
    for (cytnx_uint64 b = 0; b < B.Nblocks(); b++)
      EXPECT_TRUE(AreNearlyEqTensor(dense.get_block_(b), linalg::Diag(B.get_block_(b)), 1e-14));

    UniTensor Bp = B.clone();
    Bp.to_dense_();
    EXPECT_FALSE(Bp.is_diag());
    EXPECT_TRUE(AreEqUniTensor(Bp, dense));
  }
}

// to_dense on an already non-diagonal BlockUniTensor is a no-op (returns an equal tensor).
TEST_F(BlockUniTensorTest, to_dense_non_diag) {
  EXPECT_FALSE(BUT4.is_diag());
  EXPECT_TRUE(AreEqUniTensor(BUT4, BUT4.to_dense()));
  UniTensor Bp = BUT4.clone();
  Bp.to_dense_();
  EXPECT_TRUE(AreEqUniTensor(BUT4, Bp));
}

/*=====test info=====
describe:contraction with mixed dtypes (double lhs, float rhs); exercises
         the #ifdef UNI_MKL dtype-cast path added with Gemm_Batch
====================*/
TEST_F(BlockUniTensorTest, ContractMixedDtype) {
  UniTensor L_d = UT_contract_L1.astype(Type.Double);
  UniTensor R_f = UT_contract_R1.astype(Type.Float);
  UniTensor L_ref = UT_contract_L1.astype(Type.Double);
  UniTensor R_ref = UT_contract_R1.astype(Type.Double);
  L_d.set_labels({"a", "b"});
  R_f.set_labels({"b", "c"});
  L_ref.set_labels({"a", "b"});
  R_ref.set_labels({"b", "c"});
  UniTensor out_mixed = L_d.contract(R_f);
  UniTensor out_ref = L_ref.contract(R_ref);
  auto outbks = out_mixed.get_blocks();
  auto refbks = out_ref.get_blocks();
  for (int i = 0; i < static_cast<int>(refbks.size()); i++) {
    EXPECT_EQ(AreNearlyEqTensor(outbks[i], refbks[i], 1e-5), true);
  }
}

/*=====test info=====
describe:Integer-dtype block contractions must not throw in MKL builds.
         Gemm_Batch rejects dtype > 4; the Matmul fallback must be taken instead.
====================*/
TEST_F(BlockUniTensorTest, ContractIntegerDtype) {
  Bond bi = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 2});
  UniTensor L = UniTensor({bi, bi.redirect()}, {"a", "b"}, 1, Type.Int64, Device.cpu, false);
  UniTensor R = UniTensor({bi, bi.redirect()}, {"b", "c"}, 1, Type.Int64, Device.cpu, false);
  L.at({0, 0}) = 1;
  L.at({2, 2}) = 2;
  R.at({0, 0}) = 3;
  R.at({2, 2}) = 4;
  // Contract must not throw; result block [0,0]=1*3=3, block [2,2]=2*4=8
  UniTensor out;
  EXPECT_NO_THROW(out = Contract(L, R));
  EXPECT_EQ(int64_t(out.at({0, 0}).real()), 3);
  EXPECT_EQ(int64_t(out.at({2, 2}).real()), 8);
}

/*=====test info=====
describe:regression test for issue #724 on the contract() path. contract()
         used to permute_/reshape_ the operands' blocks in place and restore
         them afterward. When the two operands alias each other's blocks
         (relabel() is documented to share block storage), the in-place
         mutation of the left operand corrupts the right operand mid-
         contraction: the contraction throws (rank-mismatched permute_ on an
         already-reshaped shared block) or produces wrong values. contract()
         must treat both operands' blocks as read-only.
====================*/
TEST_F(BlockUniTensorTest, ContractAliasedSharedBlocksOperandsIntact) {
  Bond ba = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 2});
  Bond bb = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 3});
  Bond bc = Bond(BD_OUT, {Qs(0) >> 1, Qs(1) >> 2});
  UniTensor A = UniTensor({ba, bb, bc}, {"a", "b", "c"});
  random::uniform_(A, -1.0, 1.0, 42);
  // B shares A's blocks; contracting over "a" and "c" makes the two operands
  // of contract() alias each other's blocks (including block-with-itself pairs).
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
  for (cytnx_uint64 i = 0; i < got.Nblocks(); i++) {
    EXPECT_TRUE(AreNearlyEqTensor(got.get_blocks_()[i], expected.get_blocks_()[i], 1e-12));
  }

  // both operands must be intact: values, shapes, and contiguity
  ASSERT_EQ(A.Nblocks(), Asnap.Nblocks());
  for (cytnx_uint64 i = 0; i < A.Nblocks(); i++) {
    EXPECT_EQ(A.get_blocks_()[i].shape(), Asnap.get_blocks_()[i].shape());
    EXPECT_TRUE(AreNearlyEqTensor(A.get_blocks_()[i], Asnap.get_blocks_()[i], 0.0));
  }
  EXPECT_TRUE(A.is_contiguous());
  EXPECT_TRUE(B.is_contiguous());
}

/*=====test info=====
describe:regression test for issue #724 on the contract() path. A third
         UniTensor sharing blocks with an operand (via relabel()) must not
         observe any change from the contraction. The pre-fix mutate-and-
         restore left the shared blocks with replaced, permuted storage
         (non-contiguous), even though the values were restored.
====================*/
TEST_F(BlockUniTensorTest, ContractLeavesSharedBlockObserverUntouched) {
  Bond ba = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 2});
  Bond bb = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 3});
  Bond bc = Bond(BD_OUT, {Qs(0) >> 1, Qs(1) >> 2});
  UniTensor A = UniTensor({ba, bb, bc}, {"a", "b", "c"});
  random::uniform_(A, -1.0, 1.0, 7);
  UniTensor observer = A.relabel({"x", "y", "z"});  // shares A's blocks; not an operand
  ASSERT_TRUE(A.same_data(observer));
  UniTensor snap = A.clone();

  UniTensor R = A.clone().relabel({"c", "d", "a"});  // independent right operand
  UniTensor got = Contract(A, R);
  (void)got;

  ASSERT_EQ(observer.Nblocks(), snap.Nblocks());
  for (cytnx_uint64 i = 0; i < observer.Nblocks(); i++) {
    EXPECT_EQ(observer.get_blocks_()[i].shape(), snap.get_blocks_()[i].shape());
    EXPECT_TRUE(AreNearlyEqTensor(observer.get_blocks_()[i], snap.get_blocks_()[i], 0.0));
  }
  EXPECT_TRUE(observer.is_contiguous());
  EXPECT_TRUE(R.is_contiguous());
}
