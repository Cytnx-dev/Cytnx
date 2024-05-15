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
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 9; j++)
      for (size_t k = 1; k < 30; k++) {
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
  for (size_t i = 0; i < 10; i++)
    for (size_t j = 0; j < 6; j++)
      for (size_t k = 0; k < 10; k++) {
        EXPECT_EQ(permuted.at({i, j, k}).exists(), UT_permute_ans1.at({i, j, k}).exists());
        if (permuted.at({i, j, k}).exists())
          EXPECT_EQ(double(permuted.at({i, j, k}).real()),
                    double(UT_permute_ans1.at({i, j, k}).real()));
      }
}

TEST_F(BlockUniTensorTest, permute2) {
  std::vector<cytnx_int64> a = {1, 0};
  auto permuted = UT_permute_2.permute(a, -1);

  for (size_t j = 0; j < 10; j++)
    for (size_t k = 0; k < 10; k++) {
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
  for (size_t i = 0; i < 10; i++)
    for (size_t j = 0; j < 6; j++)
      for (size_t k = 0; k < 10; k++) {
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
  for (size_t j = 0; j < 10; j++)
    for (size_t k = 0; k < 10; k++) {
      EXPECT_EQ(permuted.at({j, k}).exists(), UT_permute_ans2.at({j, k}).exists());
      if (permuted.at({j, k}).exists())
        EXPECT_EQ(double(permuted.at({j, k}).real()), double(UT_permute_ans2.at({j, k}).real()));
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
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 9; j++)
      for (size_t k = 1; k < 30; k++) {
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
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 9; j++)
      for (size_t k = 1; k < 30; k++) {
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
  UT_pB.put_block(t0, {0, 0, 0}, true);
  UT_pB.put_block(t1a, {0, 1, 1}, true);
  UT_pB.put_block(t1b, {1, 0, 1}, true);
  UT_pB.put_block(t2, {1, 1, 2}, true);
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 9; j++)
      for (size_t k = 1; k < 30; k++) {
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
  UT_pB.put_block_(t0, {0, 0, 0}, true);
  UT_pB.put_block_(t1a, {0, 1, 1}, true);
  UT_pB.put_block_(t1b, {1, 0, 1}, true);
  UT_pB.put_block_(t2, {1, 1, 2}, true);
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 9; j++)
      for (size_t k = 1; k < 30; k++) {
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
    std::cout << outbks[i] << std::endl;
    std::cout << ansbks[i] << std::endl;
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
    std::cout << outbks[i] << std::endl;
    std::cout << ansbks[i] << std::endl;
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
    std::cout << outbks[i] << std::endl;
    std::cout << ansbks[i] << std::endl;
    EXPECT_EQ(AreNearlyEqTensor(outbks[i], ansbks[i], 1e-5), true);
  }
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
  for (size_t i = 1; i <= 5; i++)
    for (size_t j = 1; j <= 11; j++)
      for (size_t k = 1; k <= 3; k++)
        for (size_t l = 1; l <= 5; l++)
          if (out2.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(out2.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(BUTpT2.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(out2.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(BUTpT2.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
  BUT4.Add_(BUT4_2);
  for (size_t i = 1; i <= 5; i++)
    for (size_t j = 1; j <= 11; j++)
      for (size_t k = 1; k <= 3; k++)
        for (size_t l = 1; l <= 5; l++)
          if (BUT4.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(BUT4.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(BUTpT2.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(BUT4.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(BUTpT2.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
}

TEST_F(BlockUniTensorTest, Mul) {
  auto out = BUT4.Mul(9);
  for (size_t i = 1; i <= 5; i++)
    for (size_t j = 1; j <= 11; j++)
      for (size_t k = 1; k <= 3; k++)
        for (size_t l = 1; l <= 5; l++)
          if (out.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(BUTm9.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(BUTm9.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
  BUT4.Mul_(9);
  for (size_t i = 1; i <= 5; i++)
    for (size_t j = 1; j <= 11; j++)
      for (size_t k = 1; k <= 3; k++)
        for (size_t l = 1; l <= 5; l++)
          if (BUT4.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(BUT4.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(BUTm9.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(BUT4.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(BUTm9.at({i - 1, j - 1, k - 1, l - 1}).imag()));
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
  for (size_t i = 1; i <= 5; i++)
    for (size_t j = 1; j <= 11; j++)
      for (size_t k = 1; k <= 3; k++)
        for (size_t l = 1; l <= 5; l++)
          if (out2.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(out2.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(BUTsT2.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(out2.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(BUTsT2.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
  BUT4.Sub_(BUT4_2);
  for (size_t i = 1; i <= 5; i++)
    for (size_t j = 1; j <= 11; j++)
      for (size_t k = 1; k <= 3; k++)
        for (size_t l = 1; l <= 5; l++)
          if (BUT4.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(BUT4.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(BUTsT2.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(BUT4.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(BUTsT2.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
}

TEST_F(BlockUniTensorTest, Div) {
  auto out = BUT4.Div(9);
  for (size_t i = 1; i <= 5; i++)
    for (size_t j = 1; j <= 11; j++)
      for (size_t k = 1; k <= 3; k++)
        for (size_t l = 1; l <= 5; l++)
          if (out.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(BUTd9.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(BUTd9.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
  BUT4.Div_(9);
  for (size_t i = 1; i <= 5; i++)
    for (size_t j = 1; j <= 11; j++)
      for (size_t k = 1; k <= 3; k++)
        for (size_t l = 1; l <= 5; l++)
          if (BUT4.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(BUT4.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(BUTd9.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(BUT4.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(BUTd9.at({i - 1, j - 1, k - 1, l - 1}).imag()));
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

TEST_F(BlockUniTensorTest, Norm) {
  // std::cout<<BUT4<<std::endl;
  // EXPECT_TRUE(Scalar(BUT4.Norm().at({0})-10.02330912178208).abs()<1e-5);
  EXPECT_DOUBLE_EQ(double(BUT4.Norm().at({0}).real()), 10.36019459497064);

  cytnx_double tmp = double(UT_diag.Norm().at({0}).real());
  cytnx_double ans = 0;
  for (size_t i = 0; i < UT_diag.bonds()[0].qnums().size(); i++) {
    cytnx_uint64 deg = UT_diag.bonds()[0]._impl->_degs[i];
    for (int j = 0; j < deg; j++) ans += (i + 1) * (i + 1);
  }
  ans = sqrt(ans);
  EXPECT_DOUBLE_EQ(ans, tmp);
}

TEST_F(BlockUniTensorTest, Conj) {
  auto tmp = BUT4.Conj();
  for (size_t i = 1; i <= 5; i++)
    for (size_t j = 1; j <= 11; j++)
      for (size_t k = 1; k <= 3; k++)
        for (size_t l = 1; l <= 5; l++)
          if (BUT4.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            // EXPECT_TRUE(Scalar(tmp.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
            EXPECT_DOUBLE_EQ(double(tmp.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(BUT4.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(tmp.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             -double(BUT4.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
  tmp = BUT4.clone();
  BUT4.Conj_();
  for (size_t i = 1; i <= 5; i++)
    for (size_t j = 1; j <= 11; j++)
      for (size_t k = 1; k <= 3; k++)
        for (size_t l = 1; l <= 5; l++)
          if (BUT4.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            // EXPECT_TRUE(Scalar(BUT4.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
            EXPECT_DOUBLE_EQ(double(BUT4.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(tmp.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(BUT4.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             -double(tmp.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }

  tmp = UT_diag_cplx.Conj();
  for (size_t i = 0; i < UT_diag.bonds()[0].qnums().size(); i++) {
    cytnx_uint64 deg = UT_diag.bonds()[0]._impl->_degs[i];
    for (size_t j = 0; j < deg; j++) {
      EXPECT_DOUBLE_EQ(double(tmp.get_block_(i).at({j}).real()),
                       double(UT_diag_cplx.get_block_(i).at({j}).real()));
      EXPECT_DOUBLE_EQ(double(tmp.get_block_(i).at({j}).imag()),
                       -double(UT_diag_cplx.get_block_(i).at({j}).imag()));
    }
  }
}

TEST_F(BlockUniTensorTest, Transpose) {
  auto tmp = BUT1.Transpose();
  EXPECT_EQ(tmp.bonds()[0].type(), BD_OUT);
  EXPECT_EQ(tmp.bonds()[1].type(), BD_OUT);
  EXPECT_EQ(tmp.bonds()[2].type(), BD_IN);
  EXPECT_EQ(tmp.bonds()[3].type(), BD_IN);

  tmp = BUT5.Transpose();
  EXPECT_EQ(tmp.bonds()[0].type(), BD_BRA);
  EXPECT_EQ(tmp.bonds()[1].type(), BD_KET);
  EXPECT_EQ(tmp.bonds()[0].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));
  EXPECT_EQ(tmp.bonds()[1].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));

  BUT1.Transpose_();
  EXPECT_EQ(BUT1.bonds()[0].type(), BD_OUT);
  EXPECT_EQ(BUT1.bonds()[1].type(), BD_OUT);
  EXPECT_EQ(BUT1.bonds()[2].type(), BD_IN);
  EXPECT_EQ(BUT1.bonds()[3].type(), BD_IN);

  BUT5.Transpose_();
  EXPECT_EQ(BUT5.bonds()[0].type(), BD_BRA);
  EXPECT_EQ(BUT5.bonds()[1].type(), BD_KET);
  EXPECT_EQ(BUT5.bonds()[0].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));
  EXPECT_EQ(BUT5.bonds()[1].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));
}

TEST_F(BlockUniTensorTest, Trace) {
  // std::cout<<BUT4<<std::endl;
  auto tmp = BUT4.Trace(0, 3);
  // std::cout<<BUtrT4<<std::endl;
  // std::cout<<tmp<<std::endl;
  for (size_t j = 1; j <= 11; j++)
    for (size_t k = 1; k <= 3; k++)
      if (BUtrT4.at({j - 1, k - 1}).exists()) {
        // EXPECT_TRUE(Scalar(tmp.at({j-1,k-1})-BUtrT4.at({j-1,k-1})).abs()<1e-5);
        EXPECT_DOUBLE_EQ(double(tmp.at({j - 1, k - 1}).real()),
                         double(BUtrT4.at({j - 1, k - 1}).real()));
        EXPECT_DOUBLE_EQ(double(tmp.at({j - 1, k - 1}).imag()),
                         double(BUtrT4.at({j - 1, k - 1}).imag()));
      }
  // std::cout<<tmp<<std::endl;
  tmp = UT_diag.Trace(0, 1);
  cytnx_double ans = 0;
  for (size_t i = 0; i < UT_diag.bonds()[0].qnums().size(); i++) {
    cytnx_uint64 deg = UT_diag.bonds()[0]._impl->_degs[i];
    for (int j = 0; j < deg; j++) ans += i + 1;
  }
  EXPECT_DOUBLE_EQ(double(tmp.at({0}).real()), double(ans));
  EXPECT_DOUBLE_EQ(double(tmp.at({0}).imag()), double(0));

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
  EXPECT_EQ(tmp.bonds()[0].type(), BD_OUT);
  EXPECT_EQ(tmp.bonds()[1].type(), BD_OUT);
  EXPECT_EQ(tmp.bonds()[2].type(), BD_IN);
  EXPECT_EQ(tmp.bonds()[3].type(), BD_IN);

  tmp = BUT5.Dagger();
  EXPECT_EQ(tmp.bonds()[0].type(), BD_BRA);
  EXPECT_EQ(tmp.bonds()[1].type(), BD_KET);
  EXPECT_EQ(tmp.bonds()[0].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));
  EXPECT_EQ(tmp.bonds()[1].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));

  BUT1.Dagger_();
  EXPECT_EQ(BUT1.bonds()[0].type(), BD_OUT);
  EXPECT_EQ(BUT1.bonds()[1].type(), BD_OUT);
  EXPECT_EQ(BUT1.bonds()[2].type(), BD_IN);
  EXPECT_EQ(BUT1.bonds()[3].type(), BD_IN);

  BUT5.Dagger_();
  EXPECT_EQ(BUT5.bonds()[0].type(), BD_BRA);
  EXPECT_EQ(BUT5.bonds()[1].type(), BD_KET);
  EXPECT_EQ(BUT5.bonds()[0].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));
  EXPECT_EQ(BUT5.bonds()[1].qnums(),
            std::vector<std::vector<cytnx_int64>>({{0, 2}, {1, 5}, {1, 6}, {0, 1}}));

  tmp = BUT4.Dagger();
  for (size_t i = 1; i <= 5; i++)
    for (size_t j = 1; j <= 11; j++)
      for (size_t k = 1; k <= 3; k++)
        for (size_t l = 1; l <= 5; l++)
          if (BUT4.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            // EXPECT_TRUE(Scalar(tmp.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
            EXPECT_DOUBLE_EQ(double(tmp.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(BUT4.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(tmp.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             -double(BUT4.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
  tmp = BUT4.clone();
  BUT4.Dagger_();
  for (size_t i = 1; i <= 5; i++)
    for (size_t j = 1; j <= 11; j++)
      for (size_t k = 1; k <= 3; k++)
        for (size_t l = 1; l <= 5; l++)
          if (BUT4.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            // EXPECT_TRUE(Scalar(BUT4.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
            EXPECT_DOUBLE_EQ(double(BUT4.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(tmp.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(BUT4.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             -double(tmp.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }

  tmp = UT_diag_cplx.Dagger();
  for (size_t i = 0; i < UT_diag_cplx.bonds()[0].qnums().size(); i++) {
    cytnx_uint64 deg = UT_diag_cplx.bonds()[0]._impl->_degs[i];
    for (size_t j = 0; j < deg; j++) {
      EXPECT_DOUBLE_EQ(double(tmp.get_block_(i).at({j}).real()),
                       double(UT_diag_cplx.get_block_(i).at({j}).real()));
      EXPECT_DOUBLE_EQ(double(tmp.get_block_(i).at({j}).imag()),
                       -double(UT_diag_cplx.get_block_(i).at({j}).imag()));
    }
  }
}

TEST_F(BlockUniTensorTest, elem_exist) {
  for (size_t i = 1; i <= 5; i++)
    for (size_t j = 1; j <= 11; j++)
      for (size_t k = 1; k <= 3; k++)
        for (size_t l = 1; l <= 5; l++)
          if (BUT4.elem_exists({i - 1, j - 1, k - 1, l - 1})) {
            cytnx_int64 _a;
            std::vector<cytnx_uint64> _b;
            ((BlockUniTensor*)BUT4._impl.get())
              ->_fx_locate_elem(_a, _b, {i - 1, j - 1, k - 1, l - 1});
            std::vector<cytnx_uint64> qind = BUT4.get_qindices(_a);
            EXPECT_EQ(BUT4.bonds()[0].qnums()[qind[0]][0] - BUT4.bonds()[1].qnums()[qind[1]][0] +
                        BUT4.bonds()[2].qnums()[qind[2]][0] - BUT4.bonds()[3].qnums()[qind[3]][0],
                      0);
          }

  size_t offset = 0;
  for (size_t i = 0; i < UT_diag_cplx.bonds()[0].qnums().size(); i++) {
    cytnx_uint64 deg = UT_diag_cplx.bonds()[0]._impl->_degs[i];
    for (size_t j = 0; j < deg; j++) {
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
