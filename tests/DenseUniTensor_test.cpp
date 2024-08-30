#include "DenseUniTensor_test.h"
using namespace std;
using namespace cytnx;
using namespace std::complex_literals;

#include "test_tools.h"

#define FAIL_CASE_OPEN 0

TEST_F(DenseUniTensorTest, Init_by_Tensor) {
  // EXPECT_NO_THROW(dut.Init_by_Tensor(tar345, false, -1));
  // EXPECT_TRUE(utar345.same_data());
}

TEST_F(DenseUniTensorTest, Init_tagged) {
  // different types
  EXPECT_NO_THROW(
    dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.Float, Device.cpu, false, false));
  EXPECT_NO_THROW(dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.Double, Device.cpu,
                           false, false));
  EXPECT_NO_THROW(dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.ComplexFloat,
                           Device.cpu, false, false));
  EXPECT_NO_THROW(dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.ComplexDouble,
                           Device.cpu, false, false));

  // valid rowranks
  EXPECT_ANY_THROW(dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 99, Type.Float, Device.cpu,
                            false, false));
  EXPECT_NO_THROW(
    dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 2, Type.Float, Device.cpu, false, false));
  EXPECT_NO_THROW(
    dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.Float, Device.cpu, false, false));
  EXPECT_NO_THROW(dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, -1, Type.Float, Device.cpu,
                           false, false));
  EXPECT_ANY_THROW(dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, -2, Type.Float, Device.cpu,
                            false, false));
  EXPECT_NO_THROW(
    dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 0, Type.Float, Device.cpu, false, false));

  // is_diag = true, but rank>2
  EXPECT_ANY_THROW(
    dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.Float, Device.cpu, true, false));

  // is_diag = true, but rowrank!=1
  EXPECT_ANY_THROW(
    dut.Init({phy, phy.redirect()}, {"a", "b"}, 2, Type.Float, Device.cpu, true, false));

  // is_diag = true, but no outward bond
  // cout << phy << endl;
  EXPECT_ANY_THROW(dut.Init({phy, phy}, {"a", "b"}, 1, Type.Float, Device.cpu, true, false));
}

/*=====test info=====
describe:Test set_name
====================*/
TEST_F(DenseUniTensorTest, set_name) {
  ut1.set_name("org name");
  EXPECT_EQ(ut1.name(), "org name");
  std::string ut_name = "ut name";
  ut1.set_name(ut_name);
  EXPECT_EQ(ut1.name(), ut_name);
}

/*=====test info=====
describe:Test set_name uninitial unitensor
====================*/
TEST_F(DenseUniTensorTest, set_name_uninit) {
  std::string ut_name = "ut name";
  ut_uninit.set_name(ut_name);
  EXPECT_EQ(ut_uninit.name(), ut_name);
}

/*=====test info=====
describe:Test set label by index.
input:
  idx: a index to set.
  new_label: new label.
====================*/
TEST_F(DenseUniTensorTest, set_label_idx_str) {
  utzero345.set_label(1, "org_label");
  EXPECT_EQ(utzero345.labels()[1], "org_label");

  // replace with string label
  std::string new_str_label = "testing string label";
  utzero345.set_label(1, new_str_label);
  EXPECT_EQ(utzero345.labels()[1], new_str_label);
}

/*=====test info=====
describe:Test set label by replaceing the old label to new's one.
input:
  old_label: old label.
  new_label: new label.
====================*/
TEST_F(DenseUniTensorTest, set_label_old_label_replace) {
  utzero345.set_label(1, "org_label");
  EXPECT_EQ(utzero345.labels()[1], "org_label");
  std::string new_label = "testing label";
  utzero345.set_label("org_label", new_label);
  EXPECT_EQ(utzero345.labels()[1], new_label);
}

/*=====test info=====
describe:old and new label are same.
====================*/
TEST_F(DenseUniTensorTest, set_label_same) {
  std::string new_label = "testing label";
  utzero345.set_label(1, new_label);
  utzero345.set_label("testing label", new_label);
  EXPECT_EQ(utzero345.labels()[1], new_label);
}

/*=====test info=====
describe:old and new label is empty.
====================*/
TEST_F(DenseUniTensorTest, set_label_empty) {
  std::string new_label = "";
  utzero345.set_label(1, new_label);
  EXPECT_EQ(utzero345.labels()[1], new_label);
}

/*=====test info=====
describe:Test set label is out of range.
====================*/
TEST_F(DenseUniTensorTest, set_label_idx_out_of_range) {
  EXPECT_ANY_THROW(utzero345.set_label(3, "testing label"));
  std::string new_label = "testing label";
  EXPECT_ANY_THROW(utzero345.set_label(3, new_label));
}

/*=====test info=====
describe:input the old_label which is not exist.
====================*/
TEST_F(DenseUniTensorTest, set_label_duplicated) {
  std::string new_label = "testing label";
  utzero345.set_label(1, new_label);
  EXPECT_ANY_THROW(utzero345.set_label(2, new_label));
}

/*=====test info=====
describe:input duplicated label.
====================*/
TEST_F(DenseUniTensorTest, set_label_not_exist_old_label) {
  EXPECT_ANY_THROW(utzero345.set_label("Not exist label", "testing label"));
}

/*=====test info=====
describe:test set_labels.
====================*/
TEST_F(DenseUniTensorTest, set_labels_normal) {
  // vector
  std::vector<std::string> org_labels = {"org 1", "org 2", "org 3"};
  std::vector<std::string> new_labels = {"testing 1", "testing 2", "testing 3"};
  utzero345.set_labels(org_labels);
  utzero345.set_labels(new_labels);
  EXPECT_EQ(utzero345.labels(), new_labels);

  // initilizer list
  utzero345.set_labels({"org 1", "org 2", "org 3"});
  utzero345.set_labels({"testing 1", "testing 2", "testing 3"});
  EXPECT_EQ(utzero345.labels(), new_labels);
}

/*=====test info=====
describe:set_labels to uninitialized unitensor
====================*/
TEST_F(DenseUniTensorTest, set_labels_un_init) {
  std::vector<std::string> new_labels = {};
  ut_uninit.set_labels(new_labels);
  EXPECT_EQ(ut_uninit.labels(), new_labels);
}

/*=====test info=====
describe:test set_labels length not match.
====================*/
TEST_F(DenseUniTensorTest, set_labels_len_not_match) {
  // too long
  std::vector<std::string> new_labels_long = {"test1", "test2", "test3", "test4"};
  EXPECT_ANY_THROW(utzero345.set_labels(new_labels_long));
  std::vector<std::string> new_labels_short = {"test1", "test2"};
  EXPECT_ANY_THROW(utzero345.set_labels(new_labels_short));
}

/*=====test info=====
describe:test set_labels duplicated.
====================*/
TEST_F(DenseUniTensorTest, set_labels_duplicated) {
  std::vector<std::string> new_labels = {"test1", "test2", "test2", "test3"};
  EXPECT_ANY_THROW(utzero345.set_labels(new_labels));
}

TEST_F(DenseUniTensorTest, set_rowrank) {
  // Spf is a rank-3 tensor
  const auto org_rowrank = Spf.rowrank();
  for (cytnx_uint64 i = 0; i < 3; i++) {
    UniTensor ut_tmp;
    ut_tmp = Spf.set_rowrank(i);
    EXPECT_EQ(ut_tmp.rowrank(), i);
    EXPECT_EQ(Spf.rowrank(), org_rowrank);
    EXPECT_TRUE(ut_tmp.same_data(Spf));
  }
}

/*=====test info=====
describe:test set_rowrank to uninitialized UniTensor.
====================*/
TEST_F(DenseUniTensorTest, set_rowrank_un_init) {
  UniTensor ut_tmp;
  EXPECT_ANY_THROW(ut_tmp = ut_uninit.set_rowrank(0));
  EXPECT_ANY_THROW(ut_tmp = ut_uninit.set_rowrank_(0));
}

TEST_F(DenseUniTensorTest, set_rowrank_err) {
  EXPECT_ANY_THROW(Spf.set_rowrank(-2));  // set_rowrank cannot be negative!
  EXPECT_ANY_THROW(Spf.set_rowrank(-1));
  EXPECT_ANY_THROW(Spf.set_rowrank(4));  // set_rowrank can only from 0-3 for rank-3 tn
}

TEST_F(DenseUniTensorTest, set_rowrank_) {
  // Spf is a rank-3 tensor
  for (cytnx_uint64 i = 0; i < 3; i++) {
    UniTensor ut_tmp;
    ut_tmp = Spf.set_rowrank_(i);
    EXPECT_EQ(ut_tmp.rowrank(), i);
    EXPECT_EQ(Spf.rowrank(), i);
    EXPECT_TRUE(ut_tmp.same_data(Spf));
  }
}

TEST_F(DenseUniTensorTest, set_rowrank__err) {
  EXPECT_ANY_THROW(Spf.set_rowrank_(-2));  // set_rowrank cannot be negative!
  EXPECT_ANY_THROW(Spf.set_rowrank_(-1));
  EXPECT_ANY_THROW(Spf.set_rowrank_(4));  // set_rowrank can only from 0-3 for rank-3 tn
}

/*=====test info=====
describe:test Nblocks
====================*/
TEST_F(DenseUniTensorTest, Nblocks) {
  EXPECT_EQ(utzero345.Nblocks(), 1);  // dense unitensor onely 1 block
  EXPECT_EQ(ut_uninit.Nblocks(), 0);  // un-init unitensor
}

/*=====test info=====
describe:test rank
====================*/
TEST_F(DenseUniTensorTest, rank) {
  EXPECT_EQ(ut_uninit.rank(), 0);
  EXPECT_EQ(utzero345.rank(), 3);
  EXPECT_EQ(utzero3456.rank(), 4);
}

/*=====test info=====
describe:test row_rank
====================*/
TEST_F(DenseUniTensorTest, row_rank) {
  for (cytnx_uint64 i = 0; i < 3; i++) {
    auto rowrank = i;
    UniTensor ut({phy, phy.redirect(), aux}, {"1", "2", "3"}, rowrank);
    EXPECT_EQ(ut.rowrank(), rowrank);
  }
}

/*=====test info=====
describe:test dtype. Test for all possible dypte
====================*/
TEST_F(DenseUniTensorTest, dtype) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {phy, phy.redirect(), aux};
  std::vector<std::string> labels = {"1", "2", "3"};
  for (auto dtype : dtype_list) {
    auto ut = UniTensor(bonds, labels, row_rank, dtype);
    EXPECT_EQ(ut.dtype(), dtype);
  }
}

/*=====test info=====
describe:test uten_type for dense tesnor.
====================*/
TEST_F(DenseUniTensorTest, uten_type) { EXPECT_EQ(utzero345.uten_type(), UTenType.Dense); }

/*=====test info=====
describe:test uten_type for uninitialized tesnor.
====================*/
TEST_F(DenseUniTensorTest, uten_type_uninit) { EXPECT_EQ(ut_uninit.uten_type(), UTenType.Void); }

/*=====test info=====
describe:test dtype. Test for uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, dtype_uninit) { EXPECT_ANY_THROW(ut_uninit.dtype()); }

/*=====test info=====
describe:test device.
====================*/
TEST_F(DenseUniTensorTest, device) { EXPECT_EQ(Spf.device(), Device.cpu); }

/*=====test info=====
describe:test device. Test for uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, device_uninit) { EXPECT_ANY_THROW(ut_uninit.device()); }

/*=====test info=====
describe:test dtype_str. Test for all possible dypte
====================*/
TEST_F(DenseUniTensorTest, dtype_str) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {phy, phy.redirect(), aux};
  std::vector<std::string> labels = {"1", "2", "3"};
  std::vector<std::string> dtype_str_ans = {"Complex Double (Complex Float64)",
                                            "Complex Float (Complex Float32)",
                                            "Double (Float64)",
                                            "Float (Float32)",
                                            "Int64",
                                            "Uint64",
                                            "Int32",
                                            "Uint32",
                                            "Int16",
                                            "Uint16",
                                            "Bool"};
  for (size_t i = 0; i < dtype_list.size(); i++) {
    auto dtype = dtype_list[i];
    auto ut = UniTensor(bonds, labels, row_rank, dtype);
    EXPECT_EQ(ut.dtype_str(), dtype_str_ans[i]);
  }
}

/*=====test info=====
describe:test dtype_str. Test for uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, dtype_str_uninit) { EXPECT_ANY_THROW(ut_uninit.dtype_str()); }

/*=====test info=====
describe:test uten_type_str for dense tesnor.
====================*/
TEST_F(DenseUniTensorTest, uten_type_str) { EXPECT_EQ(utzero345.uten_type_str(), "Dense"); }

/*=====test info=====
describe:test uten_type for uninitialized tesnor.
====================*/
TEST_F(DenseUniTensorTest, uten_type_str_uninit) {
  EXPECT_EQ(ut_uninit.uten_type_str(), "Void (un-initialize UniTensor)");
}

TEST_F(DenseUniTensorTest, device_str) { EXPECT_EQ(Spf.device_str(), "cytnx device: CPU"); }

/*=====test info=====
describe:test device_str. Test for uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, device_str_uninit) {
  auto ut = UniTensor();  // uninitialzie
  EXPECT_ANY_THROW(ut.device_str());
}

TEST_F(DenseUniTensorTest, is_contiguous) {
  EXPECT_TRUE(Spf.is_contiguous());
  auto Spf_new = Spf.permute({2, 1, 0}, 1);
  EXPECT_FALSE(Spf_new.is_contiguous());
}

/*=====test info=====
describe:test is_diag
====================*/
TEST_F(DenseUniTensorTest, is_diag) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {phy, phy.redirect()};
  std::vector<std::string> labels = {"1", "2"};

  // default value
  auto ut_diag_default = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu);
  EXPECT_FALSE(ut_diag_default.is_diag());

  // diag false
  bool is_diag = false;
  auto ut_diag_false = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  EXPECT_FALSE(ut_diag_false.is_diag());

  // diag true
  is_diag = true;

  auto ut_diag_true = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  EXPECT_TRUE(ut_diag_true.is_diag());

  // uninitialized
  EXPECT_FALSE(ut_uninit.is_diag());
}

/*=====test info=====
describe:test is_tag
====================*/
TEST_F(DenseUniTensorTest, is_tag) {
  EXPECT_TRUE(Spf.is_tag());
  EXPECT_FALSE(utzero345.is_tag());
}

/*=====test info=====
describe:test syms for dense unitensor.
====================*/
TEST_F(DenseUniTensorTest, syms) {
  EXPECT_ANY_THROW(Spf.syms());
  EXPECT_ANY_THROW(ut_uninit.syms());
}

/*=====test info=====
describe:test is_braket_form
====================*/
TEST_F(DenseUniTensorTest, is_braket_form) {
  EXPECT_FALSE(utzero345.is_braket_form());
  EXPECT_FALSE(Spf.is_braket_form());

  // construct 1-in 1-out unitensor
  auto row_rank = 1u;
  std::vector<Bond> bonds = {phy, phy.redirect()};
  std::vector<std::string> labels = {"1", "2"};
  auto ut = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu);
  EXPECT_TRUE(ut.is_braket_form());

  // uninitialized
  EXPECT_ANY_THROW(ut_uninit.syms());
}

/*=====test info=====
describe:test labels
====================*/
TEST_F(DenseUniTensorTest, labels) {
  EXPECT_EQ(Spf.labels(), std::vector<std::string>({"1", "2", "3"}));
  EXPECT_EQ(ut_uninit.labels(), std::vector<std::string>());
}

/*=====test info=====
describe:test get_index
====================*/
TEST_F(DenseUniTensorTest, get_index) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {phy, phy.redirect()};
  std::vector<std::string> labels = {"label a", "label 2"};
  auto ut = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu);
  for (size_t idx = 0; idx < labels.size(); ++idx) {
    EXPECT_EQ(ut.get_index(labels[idx]), idx);
  }
}

/*=====test info=====
describe:test get_index, but input label not exist
====================*/
TEST_F(DenseUniTensorTest, get_index_not_exist) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {phy, phy.redirect()};
  std::vector<std::string> labels = {"label a", "label 2"};
  auto ut = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu);
  EXPECT_EQ(ut.get_index("not exist label"), -1);
}

/*=====test info=====
describe:test get_index, but input is uninitialized UniTesnor
====================*/
TEST_F(DenseUniTensorTest, get_index_uninit) { EXPECT_EQ(ut_uninit.get_index(""), -1); }

/*=====test info=====
describe:test bonds
====================*/
TEST_F(DenseUniTensorTest, bonds) {
  std::vector<Bond> bonds = {phy, phy.redirect(), aux};
  EXPECT_EQ(Spf.bonds(), bonds);
}

/*=====test info=====
describe:test bonds with empty bonds an uninitialzed UniTensor
====================*/
TEST_F(DenseUniTensorTest, bonds_empty) {
  std::vector<Bond> bonds = {};
  auto ut = UniTensor(bonds);
  EXPECT_EQ(ut.bonds(), bonds);
  EXPECT_EQ(ut_uninit.bonds(), bonds);
}

TEST_F(DenseUniTensorTest, shape) {
  EXPECT_EQ(std::vector<cytnx::cytnx_uint64>({2, 2, 1}), Spf.shape());
}

TEST_F(DenseUniTensorTest, shape_diag) {
  EXPECT_EQ(std::vector<cytnx::cytnx_uint64>({2, 2}), ut_complex_diag.shape());
  EXPECT_TRUE(ut_complex_diag.is_diag());
}

/*=====test info=====
describe:test shape with empty bonds
====================*/
TEST_F(DenseUniTensorTest, shape_empty_bonds) {
  auto ut = UniTensor(std::vector<Bond>());
  EXPECT_EQ(ut.shape(), std::vector<cytnx::cytnx_uint64>({1}));
}

/*=====test info=====
describe:test shape with uninitialzed UniTensor
====================*/
TEST_F(DenseUniTensorTest, shape_uninit) { EXPECT_ANY_THROW(ut_uninit.shape()); }

TEST_F(DenseUniTensorTest, is_blockform) {
  EXPECT_FALSE(Spf.is_blockform());
  EXPECT_FALSE(utzero345.is_blockform());
  EXPECT_ANY_THROW(ut_uninit.is_blockform());
}

TEST_F(DenseUniTensorTest, clone) {
  UniTensor cloned = ut1.clone();
  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 4; j++)
      for (size_t k = 0; k < 5; k++)
        for (size_t l = 0; l < 6; l++) {
          EXPECT_DOUBLE_EQ(double(cloned.at({i, j, k, l}).real()),
                           double(ut1.at({i, j, k, l}).real()));
          EXPECT_DOUBLE_EQ(double(cloned.at({i, j, k, l}).imag()),
                           double(ut1.at({i, j, k, l}).imag()));
        }
}

/*=====test info=====
describe:test to
====================*/
TEST_F(DenseUniTensorTest, to) {
  auto ut = Spf.to(Device.cpu);
  EXPECT_EQ(ut.device(), Device.cpu);

  // uninitialized
  EXPECT_ANY_THROW(ut_uninit.to(Device.cpu));
}

/*=====test info=====
describe:test to_
====================*/
TEST_F(DenseUniTensorTest, to_) {
  Spf.to_(Device.cpu);
  EXPECT_EQ(Spf.device(), Device.cpu);

  // uninitialized
  EXPECT_ANY_THROW(ut_uninit.to_(Device.cpu));
}

TEST_F(DenseUniTensorTest, relabels) {
  auto ut = utzero3456.relabels({"a", "b", "cd", "d"});
  EXPECT_EQ(utzero3456.labels()[0], "0");
  EXPECT_EQ(utzero3456.labels()[1], "1");
  EXPECT_EQ(utzero3456.labels()[2], "2");
  EXPECT_EQ(utzero3456.labels()[3], "3");
  EXPECT_EQ(ut.labels()[0], "a");
  EXPECT_EQ(ut.labels()[1], "b");
  EXPECT_EQ(ut.labels()[2], "cd");
  EXPECT_EQ(ut.labels()[3], "d");
  ut = utzero3456.relabels({"1", "-1", "2", "1000"});
  EXPECT_THROW(ut.relabels({"a", "a", "b", "c"}), std::logic_error);
  EXPECT_THROW(ut.relabels({"1", "1", "0", "-1"}), std::logic_error);
  EXPECT_THROW(ut.relabels({"a"}), std::logic_error);
  EXPECT_THROW(ut.relabels({"1", "2"}), std::logic_error);
  EXPECT_THROW(ut.relabels({"a", "b", "c", "d", "e"}), std::logic_error);
  EXPECT_THROW(ut_uninit.relabels({"a", "b", "c", "d", "e"}), std::logic_error);
}

TEST_F(DenseUniTensorTest, relabels_) {
  auto ut = utzero3456.relabels_({"a", "b", "cd", "d"});
  EXPECT_EQ(utzero3456.labels()[0], "a");
  EXPECT_EQ(utzero3456.labels()[1], "b");
  EXPECT_EQ(utzero3456.labels()[2], "cd");
  EXPECT_EQ(utzero3456.labels()[3], "d");
  EXPECT_EQ(ut.labels()[0], "a");
  EXPECT_EQ(ut.labels()[1], "b");
  EXPECT_EQ(ut.labels()[2], "cd");
  EXPECT_EQ(ut.labels()[3], "d");
  ut = utzero3456.relabels_({"1", "-1", "2", "1000"});
  EXPECT_THROW(ut.relabels_({"a", "a", "b", "c"}), std::logic_error);
  EXPECT_THROW(ut.relabels_({"1", "1", "0", "-1"}), std::logic_error);
  EXPECT_THROW(ut.relabels_({"a"}), std::logic_error);
  EXPECT_THROW(ut.relabels_({"1", "2"}), std::logic_error);
  EXPECT_THROW(ut.relabels_({"a", "b", "c", "d", "e"}), std::logic_error);
  EXPECT_THROW(ut_uninit.relabels_({"a", "b", "c", "d", "e"}), std::logic_error);
}

TEST_F(DenseUniTensorTest, relabel) {
  auto tmp = utzero3456.clone();
  auto ut = utzero3456.relabel({"a", "b", "cd", "d"});
  EXPECT_EQ(utzero3456.labels()[0], "0");
  EXPECT_EQ(utzero3456.labels()[1], "1");
  EXPECT_EQ(utzero3456.labels()[2], "2");
  EXPECT_EQ(utzero3456.labels()[3], "3");
  EXPECT_EQ(ut.labels()[0], "a");
  EXPECT_EQ(ut.labels()[1], "b");
  EXPECT_EQ(ut.labels()[2], "cd");
  EXPECT_EQ(ut.labels()[3], "d");
  ut = utzero3456.relabel({"1", "-1", "2", "1000"});
  EXPECT_THROW(ut.relabel({"a", "a", "b", "c"}), std::logic_error);
  EXPECT_THROW(ut.relabel({"1", "1", "0", "-1"}), std::logic_error);
  EXPECT_THROW(ut.relabel({"a"}), std::logic_error);
  EXPECT_THROW(ut.relabel({"1", "2"}), std::logic_error);
  EXPECT_THROW(ut.relabel({"a", "b", "c", "d", "e"}), std::logic_error);
  EXPECT_THROW(ut_uninit.relabel({"a", "b", "c", "d", "e"}), std::logic_error);

  utzero3456 = tmp;
  // UniTensor ut;
  ut = utzero3456.relabel("0", "a");
  EXPECT_EQ(ut.labels()[0], "a");
  ut = utzero3456.relabel("1", "b");
  ut = ut.relabel("b", "ggg");
  EXPECT_EQ(ut.labels()[1], "ggg");
  ut = utzero3456.relabel("2", "d");
  EXPECT_EQ(ut.labels()[2], "d");
  ut = utzero3456.relabel("3", "de");
  EXPECT_EQ(ut.labels()[3], "de");
  ut = utzero3456.relabel("3", "3");
  EXPECT_EQ(ut.labels()[3], "3");

  EXPECT_EQ(utzero3456.labels()[0], "0");
  EXPECT_EQ(utzero3456.labels()[1], "1");
  EXPECT_EQ(utzero3456.labels()[2], "2");
  EXPECT_EQ(utzero3456.labels()[3], "3");
  ut = utzero3456.relabel(0, "ccc");
  EXPECT_EQ(ut.labels()[0], "ccc");
  EXPECT_EQ(utzero3456.labels()[0], "0");
  ut = utzero3456.relabel(0, "-1");
  EXPECT_EQ(ut.labels()[0], "-1");
  EXPECT_EQ(utzero3456.labels()[0], "0");
  ut = utzero3456.relabel(1, "-199922");
  EXPECT_EQ(ut.labels()[1], "-199922");
  EXPECT_EQ(utzero3456.labels()[1], "1");
  ut = utzero3456.relabel(0, "-1");
  ut = ut.relabel("-1", "0");
  EXPECT_EQ(ut.labels()[0], "0");
  ut = utzero3456.relabel(0, "a").relabel("a", "a2");
  EXPECT_EQ(ut.labels()[0], "a2");

  // utzero3456.relabel(0,'a');
  // EXPECT_EQ(utzero3456.labels()[0],"a");
  EXPECT_THROW(utzero3456.relabel(5, "a"), std::logic_error);
  EXPECT_THROW(utzero3456.relabel(-1, "a"), std::logic_error);
  EXPECT_THROW(utzero3456.relabel(0, "a").relabel(1, "a"), std::logic_error);
  EXPECT_THROW(utzero3456.relabel("a", "b"), std::logic_error);
  // EXPECT_THROW(utzero3456.relabel(5,'a'),std::logic_error);
  EXPECT_THROW(ut_uninit.relabel(0, ""), std::logic_error);
}
TEST_F(DenseUniTensorTest, relabel_) {
  auto tmp = utzero3456.clone();
  auto ut = utzero3456.relabel_({"a", "b", "cd", "d"});
  EXPECT_EQ(utzero3456.labels()[0], "a");
  EXPECT_EQ(utzero3456.labels()[1], "b");
  EXPECT_EQ(utzero3456.labels()[2], "cd");
  EXPECT_EQ(utzero3456.labels()[3], "d");
  EXPECT_EQ(ut.labels()[0], "a");
  EXPECT_EQ(ut.labels()[1], "b");
  EXPECT_EQ(ut.labels()[2], "cd");
  EXPECT_EQ(ut.labels()[3], "d");
  ut = utzero3456.relabel_({"1", "-1", "2", "1000"});
  EXPECT_THROW(ut.relabel_({"a", "a", "b", "c"}), std::logic_error);
  EXPECT_THROW(ut.relabel_({"1", "1", "0", "-1"}), std::logic_error);
  EXPECT_THROW(ut.relabel_({"a"}), std::logic_error);
  EXPECT_THROW(ut.relabel_({"1", "2"}), std::logic_error);
  EXPECT_THROW(ut.relabel_({"a", "b", "c", "d", "e"}), std::logic_error);
  EXPECT_THROW(ut_uninit.relabel_({"a", "b", "c", "d", "e"}), std::logic_error);

  utzero3456 = tmp;
  // UniTensor ut;
  ut = utzero3456.relabel_("0", "a");
  ut = utzero3456.relabel_("1", "b");
  ut = utzero3456.relabel_("2", "d");
  ut = utzero3456.relabel_("3", "de");
  ut = utzero3456.relabel_("b", "ggg");
  EXPECT_EQ(utzero3456.labels()[0], "a");
  EXPECT_EQ(utzero3456.labels()[1], "ggg");
  EXPECT_EQ(utzero3456.labels()[2], "d");
  EXPECT_EQ(utzero3456.labels()[3], "de");
  EXPECT_EQ(ut.labels()[0], "a");
  EXPECT_EQ(ut.labels()[1], "ggg");
  EXPECT_EQ(ut.labels()[2], "d");
  EXPECT_EQ(ut.labels()[3], "de");
  ut = utzero3456.relabel_("de", "de");
  EXPECT_EQ(ut.labels()[3], "de");
  utzero3456.relabel_(0, "ccc");
  EXPECT_EQ(utzero3456.labels()[0], "ccc");
  utzero3456.relabel_(0, "-1");
  EXPECT_EQ(utzero3456.labels()[0], "-1");
  utzero3456.relabel_(1, "-199922");
  EXPECT_EQ(utzero3456.labels()[1], "-199922");
  utzero3456.relabel_("-1", "0");
  EXPECT_EQ(utzero3456.labels()[0], "0");
  EXPECT_THROW(utzero3456.relabel_(5, "a"), std::logic_error);
  EXPECT_THROW(utzero3456.relabel_(-1, "a"), std::logic_error);
  EXPECT_THROW(utzero3456.relabel_(0, "a").relabel_(1, "a"), std::logic_error);
  EXPECT_THROW(ut_uninit.relabel_(0, ""), std::logic_error);
}

/*=====test info=====
describe:test astype, input all possible dtype.
====================*/
TEST_F(DenseUniTensorTest, astype) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {phy, phy.redirect()};
  std::vector<std::string> labels = {"1", "2"};

  // from complex double
  auto ut_src = UniTensor(bonds, labels, row_rank, Type.ComplexDouble);
  auto ut_dst = ut_src.astype(Type.ComplexDouble);
  EXPECT_EQ(ut_dst.dtype(), Type.ComplexDouble);
  ut_dst = ut_src.astype(Type.ComplexFloat);
  EXPECT_EQ(ut_dst.dtype(), Type.ComplexFloat);
  EXPECT_THROW(ut_src.astype(Type.Double), std::logic_error);  // error test
  EXPECT_THROW(ut_src.astype(Type.Float), std::logic_error);  // error test
  EXPECT_THROW(ut_src.astype(Type.Int64), std::logic_error);  // error test
  EXPECT_THROW(ut_src.astype(Type.Uint64), std::logic_error);  // error test
  EXPECT_THROW(ut_src.astype(Type.Int32), std::logic_error);  // error test
  EXPECT_THROW(ut_src.astype(Type.Uint32), std::logic_error);  // error test
  EXPECT_THROW(ut_src.astype(Type.Int16), std::logic_error);  // error test
  EXPECT_THROW(ut_src.astype(Type.Uint16), std::logic_error);  // error test
  EXPECT_THROW(ut_src.astype(Type.Bool), std::logic_error);  // error test

  // from complex float
  ut_src = UniTensor(bonds, labels, row_rank, Type.ComplexFloat);
  ut_dst = ut_src.astype(Type.ComplexDouble);
  EXPECT_EQ(ut_dst.dtype(), Type.ComplexDouble);
  ut_dst = ut_src.astype(Type.ComplexFloat);
  EXPECT_EQ(ut_dst.dtype(), Type.ComplexFloat);
  EXPECT_THROW(ut_src.astype(Type.Double), std::logic_error);  // error test
  EXPECT_THROW(ut_src.astype(Type.Float), std::logic_error);  // error test
  EXPECT_THROW(ut_src.astype(Type.Int64), std::logic_error);  // error test
  EXPECT_THROW(ut_src.astype(Type.Uint64), std::logic_error);  // error test
  EXPECT_THROW(ut_src.astype(Type.Int32), std::logic_error);  // error test
  EXPECT_THROW(ut_src.astype(Type.Uint32), std::logic_error);  // error test
  EXPECT_THROW(ut_src.astype(Type.Int16), std::logic_error);  // error test
  EXPECT_THROW(ut_src.astype(Type.Uint16), std::logic_error);  // error test
  EXPECT_THROW(ut_src.astype(Type.Bool), std::logic_error);  // error test

  // from double
  ut_src = UniTensor(bonds, labels, row_rank, Type.Double);
  for (auto dtype_dst : dtype_list) {
    ut_dst = ut_src.astype(dtype_dst);
    EXPECT_EQ(ut_dst.dtype(), dtype_dst);
  }

  // from float
  ut_src = UniTensor(bonds, labels, row_rank, Type.Float);
  for (auto dtype_dst : dtype_list) {
    ut_dst = ut_src.astype(dtype_dst);
    EXPECT_EQ(ut_dst.dtype(), dtype_dst);
  }

  // from int64
  ut_src = UniTensor(bonds, labels, row_rank, Type.Int64);
  for (auto dtype_dst : dtype_list) {
    ut_dst = ut_src.astype(dtype_dst);
    EXPECT_EQ(ut_dst.dtype(), dtype_dst);
  }

  // from uint64
  ut_src = UniTensor(bonds, labels, row_rank, Type.Uint64);
  for (auto dtype_dst : dtype_list) {
    ut_dst = ut_src.astype(dtype_dst);
    EXPECT_EQ(ut_dst.dtype(), dtype_dst);
  }

  // from int32
  ut_src = UniTensor(bonds, labels, row_rank, Type.Int32);
  for (auto dtype_dst : dtype_list) {
    ut_dst = ut_src.astype(dtype_dst);
    EXPECT_EQ(ut_dst.dtype(), dtype_dst);
  }

  // from uint32
  ut_src = UniTensor(bonds, labels, row_rank, Type.Uint32);
  for (auto dtype_dst : dtype_list) {
    ut_dst = ut_src.astype(dtype_dst);
    EXPECT_EQ(ut_dst.dtype(), dtype_dst);
  }

  // from int16
  ut_src = UniTensor(bonds, labels, row_rank, Type.Int16);
  for (auto dtype_dst : dtype_list) {
    ut_dst = ut_src.astype(dtype_dst);
    EXPECT_EQ(ut_dst.dtype(), dtype_dst);
  }

  // from uint16
  ut_src = UniTensor(bonds, labels, row_rank, Type.Uint16);
  for (auto dtype_dst : dtype_list) {
    ut_dst = ut_src.astype(dtype_dst);
    EXPECT_EQ(ut_dst.dtype(), dtype_dst);
  }

  // from Bool
  ut_src = UniTensor(bonds, labels, row_rank, Type.Bool);
  for (auto dtype_dst : dtype_list) {
    ut_dst = ut_src.astype(dtype_dst);
    EXPECT_EQ(ut_dst.dtype(), dtype_dst);
  }
}

/*=====test info=====
describe:test permute, test all dtype
====================*/
TEST_F(DenseUniTensorTest, permute_all_dtype) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(5)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    UniTensor ut = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut, -100.0, 100.0, seed);
      ut = ut.astype(dtype);
    } else {
      ut = ut.astype(dtype);
      random::uniform_(ut, -100.0, 100.0, seed);
    }
    std::vector<cytnx_int64> map = {1, 0, 2};
    auto permuted = ut.permute(map, 1);
    EXPECT_EQ(permuted.rowrank(), 1);
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 4; j++) {
        for (size_t k = 0; k < 5; k++) {
          if (dtype == Type.ComplexDouble || dtype == Type.ComplexFloat) {
            EXPECT_DOUBLE_EQ(double(ut.at({i, j, k}).real()),
                             double(permuted.at({j, i, k}).real()));
            EXPECT_DOUBLE_EQ(double(ut.at({i, j, k}).imag()),
                             double(permuted.at({j, i, k}).imag()));
          } else {
            EXPECT_DOUBLE_EQ(double(ut.at({i, j, k}).real()),
                             double(permuted.at({j, i, k}).real()));
          }
        }
      }
    }
  }
}

/*=====test info=====
describe:test permute, as original map
====================*/
TEST_F(DenseUniTensorTest, permute_orginal_map) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(5)};
  int seed = 0;
  UniTensor ut = UniTensor(bonds);
  ut = ut.astype(Type.ComplexDouble);
  random::uniform_(ut, -100.0, 100.0, seed);
  std::vector<cytnx_int64> map = {0, 1, 2};
  auto permuted = ut.permute(map, 1);
  EXPECT_EQ(permuted.rowrank(), 1);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 5; k++) {
        EXPECT_DOUBLE_EQ(double(ut.at({i, j, k}).real()), double(permuted.at({i, j, k}).real()));
        EXPECT_DOUBLE_EQ(double(ut.at({i, j, k}).imag()), double(permuted.at({i, j, k}).imag()));
      }
    }
  }
}

/*=====test info=====
describe:test permute, test rowrank
====================*/
TEST_F(DenseUniTensorTest, permute_rowrank) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(5)};
  int seed = 0;
  UniTensor ut = UniTensor(bonds);
  ut = ut.astype(Type.ComplexDouble);
  random::uniform_(ut, -100.0, 100.0, seed);
  std::vector<cytnx_int64> map = {1, 0, 2};
  std::vector<cytnx_int64> rowranks = {0, 1, 2};
  for (auto rowrank : rowranks) {
    auto permuted = ut.permute(map, rowrank);
    EXPECT_EQ(permuted.rowrank(), rowrank);
  }
}

/*=====test info=====
describe:test permute, input diagnol UniTensor
====================*/
TEST_F(DenseUniTensorTest, permute_diag) {
  std::vector<cytnx_int64> map = {1, 0};
  cytnx_uint64 rowrank = 1;
  auto permuted = ut_complex_diag.permute(map, rowrank);
  auto shape = ut_complex_diag.shape();
  EXPECT_EQ(permuted.rowrank(), rowrank);
  for (size_t i = 0; i < shape[0]; i++) {
    EXPECT_DOUBLE_EQ(double(ut_complex_diag.at({i}).real()), double(permuted.at({i}).real()));
    EXPECT_DOUBLE_EQ(double(ut_complex_diag.at({i}).imag()), double(permuted.at({i}).imag()));
  }
}

/*=====test info=====
describe:test permute, string mapper
====================*/
TEST_F(DenseUniTensorTest, permute_str_map) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(5)};
  int seed = 0;
  UniTensor ut = UniTensor(bonds);
  ut = ut.astype(Type.ComplexDouble);
  random::uniform_(ut, -100.0, 100.0, seed);
  std::vector<std::string> map = {"2", "0", "1"};
  auto permuted = ut.permute(map, 1);
  EXPECT_EQ(permuted.rowrank(), 1);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 5; k++) {
        EXPECT_DOUBLE_EQ(double(ut.at({i, j, k}).real()), double(permuted.at({k, i, j}).real()));
        EXPECT_DOUBLE_EQ(double(ut.at({i, j, k}).imag()), double(permuted.at({k, i, j}).imag()));
      }
    }
  }
}

TEST_F(DenseUniTensorTest, permute_err) {
  EXPECT_THROW(utzero345.permute({1, 2}, 0), std::logic_error);
  EXPECT_THROW(utzero345.permute({2, 3, 1}, 0), std::logic_error);
  EXPECT_THROW(utzero345.permute({}, 0), std::logic_error);

  // for diag UniTensor, rowrank need to be 1.
  EXPECT_THROW(ut_complex_diag.permute(std::vector<cytnx_int64>({1, 0}), 0), std::logic_error);
}

/*=====test info=====
describe:test permute uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, permute_uninit) {
  EXPECT_THROW(ut_uninit.permute({}, 0), std::logic_error);
  EXPECT_THROW(ut_uninit.permute_(std::vector<std::string>(), 0), std::logic_error);
}

/*=====test info=====
describe:test permute_, test all dtype
====================*/
TEST_F(DenseUniTensorTest, permute__all_dtype) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(5)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    UniTensor ut = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut, -100.0, 100.0, seed);
      ut = ut.astype(dtype);
    } else {
      ut = ut.astype(dtype);
      random::uniform_(ut, -100.0, 100.0, seed);
    }
    std::vector<cytnx_int64> map = {1, 0, 2};
    UniTensor src = ut.clone();
    ut.permute_(map, 1);
    EXPECT_EQ(ut.rowrank(), 1);
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 4; j++) {
        for (size_t k = 0; k < 5; k++) {
          if (dtype == Type.ComplexDouble || dtype == Type.ComplexFloat) {
            EXPECT_DOUBLE_EQ(double(src.at({i, j, k}).real()), double(ut.at({j, i, k}).real()));
            EXPECT_DOUBLE_EQ(double(src.at({i, j, k}).imag()), double(ut.at({j, i, k}).imag()));
          } else {
            EXPECT_DOUBLE_EQ(double(src.at({i, j, k}).real()), double(ut.at({j, i, k}).real()));
          }
        }
      }
    }
  }
}

/*=====test info=====
describe:test permute_, as original map
====================*/
TEST_F(DenseUniTensorTest, permute__orginal_map) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(5)};
  int seed = 0;
  UniTensor ut = UniTensor(bonds);
  ut = ut.astype(Type.ComplexDouble);
  random::uniform_(ut, -100.0, 100.0, seed);
  std::vector<cytnx_int64> map = {0, 1, 2};
  auto src = ut.clone();
  ut.permute_(map, 1);
  EXPECT_EQ(ut.rowrank(), 1);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 5; k++) {
        EXPECT_DOUBLE_EQ(double(src.at({i, j, k}).real()), double(ut.at({i, j, k}).real()));
        EXPECT_DOUBLE_EQ(double(src.at({i, j, k}).imag()), double(ut.at({i, j, k}).imag()));
      }
    }
  }
}

/*=====test info=====
describe:test permute_, test rowrank
====================*/
TEST_F(DenseUniTensorTest, permute__rowrank) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(5)};
  int seed = 0;
  UniTensor ut = UniTensor(bonds);
  ut = ut.astype(Type.ComplexDouble);
  random::uniform_(ut, -100.0, 100.0, seed);
  std::vector<cytnx_int64> map = {1, 0, 2};
  std::vector<cytnx_int64> rowranks = {0, 1, 2};
  auto src = ut.clone();
  for (auto rowrank : rowranks) {
    ut = src.clone();
    ut.permute_(map, rowrank);
    EXPECT_EQ(ut.rowrank(), rowrank);
  }
}

/*=====test info=====
describe:test permute, input diagnol UniTensor
====================*/
TEST_F(DenseUniTensorTest, permute__diag) {
  std::vector<cytnx_int64> map = {1, 0};
  cytnx_uint64 rowrank = 1;
  auto src = ut_complex_diag.clone();
  ut_complex_diag.permute_(map, rowrank);
  auto shape = ut_complex_diag.shape();
  EXPECT_EQ(ut_complex_diag.rowrank(), rowrank);
  for (size_t i = 0; i < shape[0]; i++) {
    EXPECT_DOUBLE_EQ(double(ut_complex_diag.at({i}).real()), double(src.at({i}).real()));
    EXPECT_DOUBLE_EQ(double(ut_complex_diag.at({i}).imag()), double(src.at({i}).imag()));
  }
}

/*=====test info=====
describe:test permute, string mapper
====================*/
TEST_F(DenseUniTensorTest, permute__str_map) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(5)};
  int seed = 0;
  UniTensor ut = UniTensor(bonds);
  ut = ut.astype(Type.ComplexDouble);
  random::uniform_(ut, -100.0, 100.0, seed);
  std::vector<std::string> map = {"2", "0", "1"};
  auto src = ut.clone();
  ut.permute_(map, 1);
  EXPECT_EQ(ut.rowrank(), 1);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 5; k++) {
        EXPECT_DOUBLE_EQ(double(src.at({i, j, k}).real()), double(ut.at({k, i, j}).real()));
        EXPECT_DOUBLE_EQ(double(src.at({i, j, k}).imag()), double(ut.at({k, i, j}).imag()));
      }
    }
  }
}

TEST_F(DenseUniTensorTest, permute__err) {
  auto ut = utzero345.clone();
  EXPECT_THROW(ut.permute({1, 2}, 0), std::logic_error);
  EXPECT_THROW(ut.permute({2, 3, 1}, 0), std::logic_error);
  EXPECT_THROW(ut.permute({}, 0), std::logic_error);

  // for diag UniTensor, rowrank need to be 1.
  EXPECT_THROW(ut_complex_diag.permute_(std::vector<cytnx_int64>({1, 0}), 0), std::logic_error);
}

/*=====test info=====
describe:test permute_ uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, permute__uninit) {
  EXPECT_THROW(ut_uninit.permute_(std::vector<cytnx_int64>(), 0), std::logic_error);
  EXPECT_THROW(ut_uninit.permute_(std::vector<std::string>(), 0), std::logic_error);
}

TEST_F(DenseUniTensorTest, contiguous) {
  auto bk = ut1.permute({1, 3, 0, 2}).contiguous().get_block();

  int ptr = 0;
  EXPECT_TRUE(bk.is_contiguous());
  for (cytnx_uint64 i = 0; i < bk.shape()[0]; i++)
    for (cytnx_uint64 j = 0; j < bk.shape()[1]; j++)
      for (cytnx_uint64 k = 0; k < bk.shape()[2]; k++)
        for (cytnx_uint64 l = 0; l < bk.shape()[3]; l++) {
          EXPECT_EQ(complex128(bk.at({i, j, k, l})), bk.storage().at<cytnx_complex128>(ptr++));
        }
}

TEST_F(DenseUniTensorTest, contiguous_) {
  auto tmp = ut1.permute({1, 3, 0, 2});
  tmp.contiguous_();
  auto bk = tmp.get_block();
  // auto bk = ut1.permute({1,3,0,2}).contiguous().get_block();

  int ptr = 0;
  EXPECT_TRUE(bk.is_contiguous());
  for (cytnx_uint64 i = 0; i < bk.shape()[0]; i++)
    for (cytnx_uint64 j = 0; j < bk.shape()[1]; j++)
      for (cytnx_uint64 k = 0; k < bk.shape()[2]; k++)
        for (cytnx_uint64 l = 0; l < bk.shape()[3]; l++) {
          EXPECT_EQ(complex128(bk.at({i, j, k, l})), bk.storage().at<cytnx_complex128>(ptr++));
        }
}

/*=====test info=====
describe:test contiguous uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, contiguous_uninit) {
  EXPECT_THROW(ut_uninit.contiguous(), std::logic_error);
  EXPECT_THROW(ut_uninit.contiguous_(), std::logic_error);
}

/*=====test info=====
describe:test group basis
====================*/
TEST_F(DenseUniTensorTest, group_basis) {
  int seed = 0;
  random::uniform_(utzero345, -100.0, 100.0, seed);
  auto ut_grp = utzero345.group_basis();
  EXPECT_TRUE(AreEqUniTensor(utzero345, ut_grp));
}

/*=====test info=====
describe:test group basis_
====================*/
TEST_F(DenseUniTensorTest, group_basis_) {
  int seed = 0;
  random::uniform_(utzero345, -100.0, 100.0, seed);
  auto ut = utzero345.clone();
  ut.group_basis_();
  EXPECT_TRUE(AreEqUniTensor(utzero345, ut));
}

/*=====test info=====
describe:test group basis_uninit
====================*/
TEST_F(DenseUniTensorTest, group_basis_uninit) {
  EXPECT_ANY_THROW(ut_uninit.group_basis());
  EXPECT_ANY_THROW(ut_uninit.group_basis_());
}

/*=====test info=====
describe:test at
====================*/
TEST_F(DenseUniTensorTest, at) {
  auto ut_src = UniTensor({Bond(3), Bond(4), Bond(2)});
  const UniTensor cut = UniTensor({Bond(3), Bond(4), Bond(2)});
  auto loc = std::vector<cytnx_uint64>({0, 1, 0});
  for (auto dtype : dtype_list) {
    auto ut = ut_src.clone();
    switch (dtype) {
      case Type.ComplexDouble: {
        ut = ut.astype(dtype);
        auto elem = complex<double>(1, -1);
        ut.at(loc) = elem;
        EXPECT_EQ(ut.at(loc), elem);
        EXPECT_EQ(ut.at<complex<double>>(loc), elem);
        EXPECT_EQ(cut.at(loc), complex<double>());
        EXPECT_EQ(cut.at<complex<double>>(loc), complex<double>());
      } break;
      case Type.ComplexFloat: {
        ut = ut.astype(dtype);
        auto elem = complex<float>(1, -1);
        ut.at(loc) = elem;
        EXPECT_EQ(ut.at(loc), elem);
        EXPECT_EQ(ut.at<complex<float>>(loc), elem);
        EXPECT_EQ(cut.at(loc), complex<float>());
        EXPECT_EQ(cut.at<complex<float>>(loc), complex<float>());
      } break;
      case Type.Double: {
        ut = ut.astype(dtype);
        auto elem = double(1);
        ut.at(loc) = elem;
        EXPECT_EQ(ut.at(loc), elem);
        EXPECT_EQ(ut.at<double>(loc), elem);
        EXPECT_EQ(cut.at(loc), double());
        EXPECT_EQ(cut.at<double>(loc), double());
      } break;
      case Type.Float: {
        ut = ut.astype(dtype);
        auto elem = float(1);
        ut.at(loc) = elem;
        EXPECT_EQ(ut.at(loc), elem);
        EXPECT_EQ(ut.at<float>(loc), elem);
        EXPECT_EQ(cut.at(loc), float());
        EXPECT_EQ(cut.at<float>(loc), float());
      } break;
      case Type.Int64: {
        ut = ut.astype(dtype);
        auto elem = cytnx_int64(1);
        ut.at(loc) = elem;
        EXPECT_EQ(ut.at(loc), elem);
        EXPECT_EQ(ut.at<cytnx_int64>(loc), elem);
        EXPECT_EQ(cut.at(loc), cytnx_int64());
        EXPECT_EQ(cut.at<cytnx_int64>(loc), cytnx_int64());
      } break;
      case Type.Uint64: {
        ut = ut.astype(dtype);
        auto elem = cytnx_uint64(1);
        ut.at(loc) = elem;
        EXPECT_EQ(ut.at(loc), elem);
        EXPECT_EQ(ut.at<cytnx_uint64>(loc), elem);
        EXPECT_EQ(cut.at(loc), cytnx_uint64());
        EXPECT_EQ(cut.at<cytnx_uint64>(loc), cytnx_uint64());
      } break;
      case Type.Int32: {
        ut = ut.astype(dtype);
        auto elem = cytnx_int32(1);
        ut.at(loc) = elem;
        EXPECT_EQ(ut.at(loc), elem);
        EXPECT_EQ(ut.at<cytnx_int32>(loc), elem);
        EXPECT_EQ(cut.at(loc), cytnx_int32());
        EXPECT_EQ(cut.at<cytnx_int32>(loc), cytnx_int32());
      } break;
      case Type.Uint32: {
        ut = ut.astype(dtype);
        auto elem = cytnx_uint32(1);
        ut.at(loc) = elem;
        EXPECT_EQ(ut.at(loc), elem);
        EXPECT_EQ(ut.at<cytnx_uint32>(loc), elem);
        EXPECT_EQ(cut.at(loc), cytnx_uint32());
        EXPECT_EQ(cut.at<cytnx_uint32>(loc), cytnx_uint32());
      } break;
      case Type.Int16: {
        ut = ut.astype(dtype);
        auto elem = cytnx_int16(1);
        ut.at(loc) = elem;
        EXPECT_EQ(ut.at(loc), elem);
        EXPECT_EQ(ut.at<cytnx_int16>(loc), elem);
        EXPECT_EQ(cut.at(loc), cytnx_int16());
        EXPECT_EQ(cut.at<cytnx_int16>(loc), cytnx_int16());
      } break;
      case Type.Uint16: {
        ut = ut.astype(dtype);
        auto elem = cytnx_uint16(1);
        ut.at(loc) = elem;
        EXPECT_EQ(ut.at(loc), elem);
        EXPECT_EQ(ut.at<cytnx_uint16>(loc), elem);
        EXPECT_EQ(cut.at(loc), cytnx_uint16());
        EXPECT_EQ(cut.at<cytnx_uint16>(loc), cytnx_uint16());
      } break;
      case Type.Bool: {
        ut = ut.astype(dtype);
        auto elem = true;
        ut.at(loc) = elem;
        EXPECT_EQ(ut.at(loc), elem);
        EXPECT_EQ(cut.at(loc), bool());
      } break;
      default:
        ASSERT_TRUE(false);
    }
  }
}

/*=====test info=====
describe:test get_block with default index
====================*/
TEST_F(DenseUniTensorTest, get_block_default_idx) {
  auto tens = zeros({2, 4});
  int seed = 0;
  random::uniform_(tens, -100.0, 100.0, seed);
  auto ut_src = UniTensor(tens);
  auto bk = ut_src.get_block();
  EXPECT_TRUE(AreEqTensor(bk, tens));
  double elem = 104.0;
  auto loc = std::vector<cytnx_uint64>({0, 0});
  double src_elem = bk.at<double>(loc);
  bk.at(loc) = elem;
  EXPECT_EQ(bk.at(loc), elem);
  EXPECT_EQ(ut_src.at(loc), src_elem);
}

/*=====test info=====
describe:test get_block
====================*/
TEST_F(DenseUniTensorTest, get_block) {
  auto tens = zeros({2, 4});
  int seed = 0;
  random::uniform_(tens, -100.0, 100.0, seed);
  auto ut_src = UniTensor(tens);
  auto bk = ut_src.get_block();
  auto bk0 = ut_src.get_block(0);
  EXPECT_TRUE(AreEqTensor(bk, bk0));
}

/*=====test info=====
describe:test get_block out of range
====================*/
#if FAIL_CASE_OPEN
TEST_F(DenseUniTensorTest, get_block_out_of_range) {
  EXPECT_THROW(utzero345.get_block(3), std::logic_error);
}
#endif

/*=====test info=====
describe:test get_block, diagonal
====================*/
TEST_F(DenseUniTensorTest, get_block_diag) {
  auto tens = zeros({4});
  int seed = 0;
  random::uniform_(tens, -100.0, 100.0, seed);
  bool is_diag = true;
  auto ut_src = UniTensor(tens, is_diag);
  auto bk = ut_src.get_block();
  EXPECT_TRUE(AreEqTensor(bk, tens));
  double elem = 104.0;
  auto loc = std::vector<cytnx_uint64>({1});
  double src_elem = bk.at<double>(loc);
  bk.at(loc) = elem;
  EXPECT_EQ(bk.at(loc), elem);
  EXPECT_EQ(ut_src.at(loc), src_elem);
}

/*=====test info=====
describe:test get_block with qindex
====================*/
TEST_F(DenseUniTensorTest, get_block_q) {
  auto qidx = std::vector<cytnx_int64>({1, 2});
  auto qidx2 = std::vector<cytnx_uint64>({1, 2});
  EXPECT_THROW(utzero345.get_block(qidx), std::logic_error);
  EXPECT_THROW(utzero345.get_block(qidx2), std::logic_error);
  EXPECT_THROW(utzero345.get_block({1, 2}), std::logic_error);
}

/*=====test info=====
describe:test get_block, input uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, get_block_uninit) {
  EXPECT_THROW(ut_uninit.get_block({1, 2}), std::logic_error);
}

/*=====test info=====
describe:test get_block_ with default index
====================*/
TEST_F(DenseUniTensorTest, get_block__default_idx) {
  auto tens = zeros({2, 4});
  int seed = 0;
  random::uniform_(tens, -100.0, 100.0, seed);
  auto ut_src = UniTensor(tens);
  auto bk = ut_src.get_block_();
  EXPECT_TRUE(AreEqTensor(bk, tens));
  double elem = 104.0;
  auto loc = std::vector<cytnx_uint64>({0, 0});
  bk.at(loc) = elem;
  EXPECT_EQ(bk.at(loc), elem);
  EXPECT_EQ(ut_src.at(loc), elem);
}

/*=====test info=====
describe:test get_block_
====================*/
TEST_F(DenseUniTensorTest, get_block_) {
  auto tens = zeros({2, 4});
  int seed = 0;
  random::uniform_(tens, -100.0, 100.0, seed);
  auto ut_src = UniTensor(tens);
  const UniTensor cut_src = UniTensor(tens).clone();
  auto bk = ut_src.get_block_();
  auto bk0 = ut_src.get_block_(0);
  auto bkc = cut_src.get_block_(0);
  EXPECT_TRUE(AreEqTensor(bk, bk0));
  EXPECT_TRUE(AreEqTensor(bk, bkc));
}

/*=====test info=====
describe:test get_block_, diagonal
====================*/
TEST_F(DenseUniTensorTest, get_block__diag) {
  auto tens = zeros({4});
  int seed = 0;
  random::uniform_(tens, -100.0, 100.0, seed);
  bool is_diag = true;
  auto ut_src = UniTensor(tens, is_diag);
  auto bk = ut_src.get_block_();
  EXPECT_TRUE(AreEqTensor(bk, tens));
  double elem = 104.0;
  auto loc = std::vector<cytnx_uint64>({1});
  bk.at(loc) = elem;
  EXPECT_EQ(bk.at(loc), elem);
  EXPECT_EQ(ut_src.at(loc), elem);
}

/*=====test info=====
describe:test get_block_ with qindex
====================*/
TEST_F(DenseUniTensorTest, get_block__q) {
  auto qidx = std::vector<cytnx_int64>({1, 2});
  auto qidx2 = std::vector<cytnx_uint64>({1, 2});
  const UniTensor cut = utzero345.clone();
  EXPECT_THROW(utzero345.get_block_(qidx), std::logic_error);
  EXPECT_THROW(utzero345.get_block_(qidx2), std::logic_error);
  EXPECT_THROW(utzero345.get_block_({1, 2}), std::logic_error);
  EXPECT_THROW(cut.get_block_(qidx), std::logic_error);
  EXPECT_THROW(cut.get_block_(qidx2), std::logic_error);
  EXPECT_THROW(cut.get_block_({1, 2}), std::logic_error);
}

/*=====test info=====
describe:test get_block_, input uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, get_block__uninit) {
  EXPECT_THROW(ut_uninit.get_block(), std::logic_error);
  EXPECT_THROW(ut_uninit.get_block_(), std::logic_error);
}

/*=====test info=====
describe:test get_block out of range
====================*/
#if FAIL_CASE_OPEN
TEST_F(DenseUniTensorTest, get_block__out_of_range) {
  EXPECT_THROW(utzero345.get_block_(3), std::logic_error);
}
#endif

TEST_F(DenseUniTensorTest, get_blocks) { EXPECT_THROW(utzero345.get_blocks(), std::logic_error); }

TEST_F(DenseUniTensorTest, get_blocks_) {
  const UniTensor cut = utzero345.clone();
  EXPECT_THROW(utzero345.get_blocks_(), std::logic_error);
  EXPECT_THROW(cut.get_blocks_(), std::logic_error);
}

/*=====test info=====
describe:test get_blocks, input uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, get_blocks_uninit) {
  EXPECT_THROW(ut_uninit.get_blocks(), std::logic_error);
  EXPECT_THROW(ut_uninit.get_blocks_(), std::logic_error);
}

/*=====test info=====
describe:test put_block
====================*/
TEST_F(DenseUniTensorTest, put_block) {
  constexpr cytnx_uint64 dim1 = 2, dim2 = 3;
  auto tens = zeros({dim1, dim2});
  int seed = 0;
  random::uniform_(tens, -100.0, 100.0, seed);
  auto loc = std::vector<cytnx_uint64>({1, 2});
  double src_elem = tens.at<double>(loc);
  auto ut = UniTensor({Bond(dim1), Bond(dim2)});

  ut.put_block(tens);
  EXPECT_TRUE(AreEqTensor(ut.get_block_(), tens));
  double elem = 104.0;
  tens.at(loc) = elem;
  EXPECT_EQ(tens.at(loc), elem);
  EXPECT_EQ(ut.at(loc), src_elem);

  // test for there are some data in original UniTensor
  ut.put_block(tens);
  EXPECT_TRUE(AreEqTensor(ut.get_block_(), tens));
}

/*=====test info=====
describe:test put_block, type different
UniTensor:double type unitensor
Tensor:bool type unitensor
result:after putblock, the UniTenosr becomes bool type
====================*/
TEST_F(DenseUniTensorTest, put_block_diff_type) {
  constexpr cytnx_uint64 dim1 = 2, dim2 = 3;
  auto tens = zeros({dim1, dim2}, Type.Bool);
  auto ut = UniTensor({Bond(dim1), Bond(dim2)});
  EXPECT_EQ(tens.dtype(), Type.Bool);
  EXPECT_EQ(ut.dtype(), Type.Double);
  ut.put_block(tens);
  EXPECT_EQ(ut.dtype(), Type.Bool);
  EXPECT_TRUE(AreEqTensor(ut.get_block_(), tens));
}

/*=====test info=====
describe:test put_block, diagonal
====================*/
TEST_F(DenseUniTensorTest, put_block_diag) {
  constexpr cytnx_uint64 dim1 = 3, dim2 = 3;
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(dim1), Bond(dim2)};
  std::vector<std::string> labels = {"1", "2"};
  bool is_diag = true;
  auto ut = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  auto tens = zeros({dim1});
  int seed = 0;
  random::uniform_(tens, -100.0, 100.0, seed);

  auto loc = std::vector<cytnx_uint64>({1});
  double src_elem = tens.at<double>(loc);

  ut.put_block(tens);
  EXPECT_TRUE(AreEqTensor(ut.get_block_(), tens));
  double elem = 104.0;
  tens.at(loc) = elem;
  EXPECT_EQ(tens.at(loc), elem);
  EXPECT_EQ(ut.at(loc), src_elem);

  // put block again
  ut.put_block(tens);
  EXPECT_TRUE(AreEqTensor(ut.get_block_(), tens));
  EXPECT_TRUE(ut.is_diag());
}

/*=====test info=====
describe:test put_block, rank mismatch
====================*/
TEST_F(DenseUniTensorTest, put_block_rank_mismatch) {
  constexpr cytnx_uint64 dim1 = 2, dim2 = 3;
  auto tens = zeros({dim1, dim2});
  int seed = 0;
  random::uniform_(tens, -100.0, 100.0, seed);
  auto loc = std::vector<cytnx_uint64>({1, 2});
  double src_elem = tens.at<double>(loc);
  auto ut = UniTensor({Bond(dim1 + 1), Bond(dim2)});
  EXPECT_THROW(ut.put_block(tens), std::logic_error);
}

/*=====test info=====
describe:test put_block_, out of index
====================*/
#if FAIL_CASE_OPEN
TEST_F(DenseUniTensorTest, put_block_out_of_idx) {
  constexpr cytnx_uint64 dim1 = 2, dim2 = 3;
  auto tens = zeros({dim1, dim2});
  int seed = 0;
  random::uniform_(tens, -100.0, 100.0, seed);
  auto loc = std::vector<cytnx_uint64>({1, 2});
  double src_elem = tens.at<double>(loc);
  auto ut = UniTensor({Bond(dim1), Bond(dim2)});
  EXPECT_THROW(ut.put_block(tens, 1), std::logic_error);
}
#endif

/*=====test info=====
describe:test put_block_
====================*/
TEST_F(DenseUniTensorTest, put_block_) {
  constexpr cytnx_uint64 dim1 = 2, dim2 = 3;
  auto tens = zeros({dim1, dim2});
  int seed = 0;
  random::uniform_(tens, -100.0, 100.0, seed);
  auto loc = std::vector<cytnx_uint64>({1, 2});
  auto ut = UniTensor({Bond(dim1), Bond(dim2)});
  ut.put_block_(tens);
  EXPECT_TRUE(AreEqTensor(ut.get_block_(), tens));
  double elem = 104.0;
  tens.at(loc) = elem;
  EXPECT_EQ(tens.at(loc), elem);
  EXPECT_EQ(ut.at(loc), elem);

  // put block again
  ut.put_block_(tens);
  EXPECT_TRUE(AreEqTensor(ut.get_block_(), tens));
}

/*=====test info=====
describe:test put_block_, type different
UniTensor:double type unitensor
Tensor:bool type unitensor
result:after put_block_, the UniTenosr becomes bool type
====================*/
TEST_F(DenseUniTensorTest, put_block__diff_type) {
  constexpr cytnx_uint64 dim1 = 2, dim2 = 3;
  auto tens = zeros({dim1, dim2}, Type.Bool);
  auto ut = UniTensor({Bond(dim1), Bond(dim2)});
  EXPECT_EQ(tens.dtype(), Type.Bool);
  EXPECT_EQ(ut.dtype(), Type.Double);
  ut.put_block_(tens);
  EXPECT_EQ(ut.dtype(), Type.Bool);
  EXPECT_TRUE(AreEqTensor(ut.get_block_(), tens));
}

/*=====test info=====
describe:test put_block_, diagonal
====================*/
TEST_F(DenseUniTensorTest, put_block__diag) {
  constexpr cytnx_uint64 dim1 = 3, dim2 = 3;
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(dim1), Bond(dim2)};
  std::vector<std::string> labels = {"1", "2"};
  bool is_diag = true;
  auto ut = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  auto tens = zeros({dim1});
  int seed = 0;
  random::uniform_(tens, -100.0, 100.0, seed);

  auto loc = std::vector<cytnx_uint64>({1});
  ut.put_block_(tens);
  EXPECT_TRUE(AreEqTensor(ut.get_block_(), tens));
  double elem = 104.0;
  tens.at(loc) = elem;
  EXPECT_EQ(tens.at(loc), elem);
  EXPECT_EQ(ut.at(loc), elem);

  ut.put_block(tens);
  EXPECT_TRUE(AreEqTensor(ut.get_block_(), tens));
  EXPECT_TRUE(ut.is_diag());
}

/*=====test info=====
describe:test put_block_, rank mismatch
====================*/
TEST_F(DenseUniTensorTest, put_block__rank_mismatch) {
  constexpr cytnx_uint64 dim1 = 2, dim2 = 3;
  auto tens = zeros({dim1, dim2});
  int seed = 0;
  random::uniform_(tens, -100.0, 100.0, seed);
  auto loc = std::vector<cytnx_uint64>({1, 2});
  double src_elem = tens.at<double>(loc);
  auto ut = UniTensor({Bond(dim1 + 1), Bond(dim2)});
  EXPECT_THROW(ut.put_block_(tens), std::logic_error);
}

/*=====test info=====
describe:test put_block_, out of index
====================*/
#if FAIL_CASE_OPEN
TEST_F(DenseUniTensorTest, put_block__out_of_idx) {
  constexpr cytnx_uint64 dim1 = 2, dim2 = 3;
  auto tens = zeros({dim1, dim2});
  int seed = 0;
  random::uniform_(tens, -100.0, 100.0, seed);
  auto loc = std::vector<cytnx_uint64>({1, 2});
  double src_elem = tens.at<double>(loc);
  auto ut = UniTensor({Bond(dim1), Bond(dim2)});
  EXPECT_THROW(ut.put_block_(tens, 1), std::logic_error);
}
#endif

/*=====test info=====
describe:test put_blocks, input uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, put_block_uninit) {
  EXPECT_THROW(ut_uninit.put_block(tzero345), std::logic_error);
  EXPECT_THROW(ut_uninit.put_block_(tzero345), std::logic_error);
}

/*=====test info=====
describe:test reshape
====================*/
TEST_F(DenseUniTensorTest, reshape) {
  auto src_ut = UniTensor({Bond(3), Bond(4), Bond(5)});
  int seed = 0;
  random::uniform_(src_ut, -100.0, 100.0, seed);
  const auto src_shape = src_ut.shape();
  std::vector<cytnx_int64> dst_shape = {6, 10};
  auto dst_ut = src_ut.reshape(dst_shape);
  EXPECT_EQ(src_ut.shape(), src_shape);
  // reshape input cytnx_int64 but shape return cytnx_uint64
  std::vector<cytnx_uint64> tmp;
  for (auto dim : dst_shape) tmp.push_back(static_cast<cytnx_uint64>(dim));
  EXPECT_EQ(dst_ut.shape(), tmp);
  EXPECT_EQ(dst_ut.rowrank(), 0);
  for (cytnx_uint64 i = 0; i < src_shape[0]; i++) {
    for (cytnx_uint64 j = 0; j < src_shape[1]; j++) {
      for (cytnx_uint64 k = 0; k < src_shape[2]; k++) {
        int idx = i * src_shape[1] * src_shape[2] + j * src_shape[2] + k;
        cytnx_uint64 dst_idx0 = idx / dst_shape[1];
        cytnx_uint64 dst_idx1 = idx % dst_shape[1];
        EXPECT_EQ(src_ut.at({i, j, k}), dst_ut.at({dst_idx0, dst_idx1}));
      }
    }
  }
}

/*=====test info=====
describe:test reshape, but shape not change
====================*/
TEST_F(DenseUniTensorTest, reshape_not_change) {
  auto src_ut = UniTensor({Bond(3), Bond(4), Bond(5)});
  int seed = 0;
  random::uniform_(src_ut, -100.0, 100.0, seed);
  const auto src_shape = src_ut.shape();
  std::vector<cytnx_int64> dst_shape = {3, 4, 5};
  auto dst_ut = src_ut.reshape(dst_shape);
  EXPECT_EQ(src_ut.shape(), src_shape);
  EXPECT_EQ(dst_ut.rowrank(), 0);
  // reshape input cytnx_int64 but shape return cytnx_uint64
  std::vector<cytnx_uint64> tmp;
  for (auto dim : dst_shape) tmp.push_back(static_cast<cytnx_uint64>(dim));
  EXPECT_EQ(dst_ut.shape(), tmp);
}

/*=====test info=====
describe:test reshape with digonal UniTensor
UniTensor:diagonal UniTensor
shape:(4, 4)->(2, 8)
result:return non diagonal UniTensor and correct shape
====================*/
TEST_F(DenseUniTensorTest, reshape_diag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  bool is_diag = true;
  auto ut_diag = UniTensor({Bond(4), Bond(4)}, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut_diag, -100.0, 100.0, seed);
  auto dst_ut = ut_diag.reshape({2, 8});
  EXPECT_TRUE(ut_diag.is_diag());
  EXPECT_FALSE(dst_ut.is_diag());
  auto src_shape = ut_diag.shape();
  auto dst_shape = dst_ut.shape();
  EXPECT_EQ(dst_ut.shape(), std::vector<cytnx_uint64>({2, 8}));
  EXPECT_EQ(dst_ut.rowrank(), 0);
  for (cytnx_uint64 i = 0; i < src_shape[0]; ++i) {
    for (cytnx_uint64 j = 0; j < src_shape[1]; ++j) {
      int idx = i * src_shape[1] + j;
      cytnx_uint64 dst_idx0 = idx / dst_shape[1];
      cytnx_uint64 dst_idx1 = idx % dst_shape[1];
      if (i == j) {
        EXPECT_EQ(ut_diag.at({i}), dst_ut.at({dst_idx0, dst_idx1}));
      } else {
        EXPECT_EQ(dst_ut.at({dst_idx0, dst_idx1}), 0.0);
      }
    }
  }
}

/*=====test info=====
describe:test reshape with digonal UniTensor but shape not change
UniTensor:diagonal UniTensor
shape:(4, 4)->(4, 4)
result:return non diagonal UniTensor and correct shape
====================*/
TEST_F(DenseUniTensorTest, reshape_diag_not_change) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  bool is_diag = true;
  auto ut_diag = UniTensor({Bond(4), Bond(4)}, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut_diag, -100.0, 100.0, seed);
  auto dst_ut = ut_diag.reshape({4, 4}, 1);
  EXPECT_TRUE(ut_diag.is_diag());
  EXPECT_FALSE(dst_ut.is_diag());
  EXPECT_EQ(dst_ut.shape(), ut_diag.shape());
  EXPECT_EQ(dst_ut.rowrank(), 1);
  auto src_shape = ut_diag.shape();
  auto dst_shape = dst_ut.shape();
  for (cytnx_uint64 i = 0; i < src_shape[0]; ++i) {
    for (cytnx_uint64 j = 0; j < src_shape[1]; ++j) {
      int idx = i * src_shape[1] + j;
      cytnx_uint64 dst_idx0 = idx / dst_shape[1];
      cytnx_uint64 dst_idx1 = idx % dst_shape[1];
      if (i == j) {
        EXPECT_EQ(ut_diag.at({i}), dst_ut.at({dst_idx0, dst_idx1}));
      } else {
        EXPECT_EQ(dst_ut.at({dst_idx0, dst_idx1}), 0.0);
      }
    }
  }
}

/*=====test info=====
describe:test reshape_
====================*/
TEST_F(DenseUniTensorTest, reshape_) {
  auto src_ut = UniTensor({Bond(3), Bond(4), Bond(5)});
  int seed = 0;
  random::uniform_(src_ut, -100.0, 100.0, seed);
  const auto src_shape = src_ut.shape();
  std::vector<cytnx_int64> dst_shape = {6, 10};
  auto dst_ut1 = src_ut.reshape(dst_shape);
  auto dst_ut2 = src_ut.clone();
  dst_ut2.reshape_(dst_shape);
  EXPECT_TRUE(AreEqUniTensor(dst_ut1, dst_ut2));
  EXPECT_EQ(dst_ut2.shape(), std::vector<cytnx_uint64>({6, 10}));
}

/*=====test info=====
describe:test reshape_ with diagonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, reshape__diag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  bool is_diag = true;
  auto ut_diag = UniTensor({Bond(4), Bond(4)}, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut_diag, -100.0, 100.0, seed);
  std::vector<cytnx_int64> dst_shape = {4, 4};
  auto dst_ut1 = ut_diag.reshape(dst_shape, 1);
  auto dst_ut2 = ut_diag.clone();
  dst_ut2.reshape_(dst_shape, 1);
  EXPECT_TRUE(AreEqUniTensor(dst_ut1, dst_ut2));
  EXPECT_EQ(dst_ut2.shape(), std::vector<cytnx_uint64>({4, 4}));
}

/*=====test info=====
describe:error test for reshape
====================*/
TEST_F(DenseUniTensorTest, reshape_error) {
  // shape mismatch
  EXPECT_THROW(utzero345.reshape({2, 1, 4}), std::logic_error);
  EXPECT_THROW(utzero345.reshape({}), std::logic_error);
  // rowrank too large
  auto rowrank = 4u;
  EXPECT_THROW(utzero345.reshape({3, 4, 5}, rowrank), std::logic_error);
}

/*=====test info=====
describe:error test for reshape_
====================*/
TEST_F(DenseUniTensorTest, reshape__error) {
  // shape mismatch
  EXPECT_THROW(utzero345.reshape_({2, 1, 4}), std::logic_error);
  EXPECT_THROW(utzero345.reshape_({}), std::logic_error);
  // rowrank too large
  auto rowrank = 4u;
  EXPECT_THROW(utzero345.reshape_({3, 4, 5}, rowrank), std::logic_error);
}

/*=====test info=====
describe:test reshape with uninitialized UniTensor.
====================*/
TEST_F(DenseUniTensorTest, reshape_utuninit) {
  EXPECT_ANY_THROW(ut_uninit.reshape({}));
  EXPECT_ANY_THROW(ut_uninit.reshape_({}));
}

/*=====test info=====
describe:test to_dense
====================*/
TEST_F(DenseUniTensorTest, to_dense) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  bool is_diag = true;
  auto ut_diag = UniTensor({Bond(4), Bond(4)}, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut_diag, -100.0, 100.0, seed);
  auto dst_ut = ut_diag.to_dense();
  EXPECT_TRUE(ut_diag.is_diag());
  EXPECT_FALSE(dst_ut.is_diag());
  EXPECT_EQ(dst_ut.rowrank(), row_rank);
  auto src_shape = ut_diag.shape();
  for (cytnx_uint64 i = 0; i < src_shape[0]; ++i) {
    for (cytnx_uint64 j = 0; j < src_shape[1]; ++j) {
      if (i == j) {
        EXPECT_EQ(ut_diag.at({i}), dst_ut.at({i, i}));
      } else {
        EXPECT_EQ(dst_ut.at({i, j}), 0.0);
      }
    }
  }
}

/*=====test info=====
describe:test to_dense, but the UniTensor is non diagonal
====================*/
TEST_F(DenseUniTensorTest, to_dense_non_diag) {
  auto ut_diag = UniTensor({Bond(4), Bond(4)});
  int seed = 0;
  random::uniform_(ut_diag, -100.0, 100.0, seed);
  EXPECT_THROW(ut_diag.to_dense(), std::logic_error);
}

/*=====test info=====
describe:test to_dense_, but the UniTensor is non diagonal
====================*/
TEST_F(DenseUniTensorTest, to_dense__non_diag) {
  auto ut_diag = UniTensor({Bond(4), Bond(4)});
  EXPECT_THROW(ut_diag.to_dense_(), std::logic_error);
}

/*=====test info=====
describe:test to_dense with uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, to_dense_ut_uninit) {
  EXPECT_ANY_THROW(ut_uninit.to_dense());
  EXPECT_ANY_THROW(ut_uninit.to_dense_());
}

/*=====test info=====
describe:test combineBonds
====================*/
TEST_F(DenseUniTensorTest, combineBonds) {
  std::vector<std::string> labels = {"a", "b", "c"};
  auto ut = UniTensor({Bond(5), Bond(4), Bond(3)}, labels);
  ut.set_rowrank(1);
  int seed = 0;
  random::uniform_(ut, -100.0, 100.0, seed);
  std::vector<std::string> labels_combine = {"b", "c"};
  ut.combineBonds(labels_combine);

  // construct answer directly
  labels = {"a", "b"};
  int rowrank = 1;
  auto ans_ut = UniTensor({Bond(5), Bond(12)}, labels, rowrank);
  auto tens = ut.get_block().reshape({5, 12});
  ans_ut.put_block(tens);

  // compare
  EXPECT_TRUE(AreEqUniTensor(ut, ans_ut));
}

/*=====test info=====
describe:test combineBonds with diagonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, combinebonds_diag) {
  EXPECT_THROW(ut_complex_diag.combineBonds(ut_complex_diag.labels()), std::logic_error);
}

/*=====test info=====
describe:test combineBonds error
====================*/
TEST_F(DenseUniTensorTest, combinebonds_error) {
  std::vector<std::string> labels = {"a", "b", "c"};
  auto ut = UniTensor({Bond(5), Bond(4), Bond(3)}, labels);
  ut.set_rowrank(1);
  int seed = 0;
  random::uniform_(ut, -100.0, 100.0, seed);

  // not exist labels
  std::vector<std::string> labels_combine = {"c", "d"};
  EXPECT_THROW(ut.combineBonds(labels_combine), std::logic_error);

  // empty combine's label
  labels_combine = std::vector<std::string>();
  EXPECT_THROW(ut.combineBonds(labels_combine), std::logic_error);
}

/*=====test info=====
describe:test combindbonds with uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, combineBonds_ut_uninit) {
  std::vector<std::string> labels_combine = {};
  EXPECT_ANY_THROW(ut_uninit.combineBonds(labels_combine));
}

/*=====test info=====
describe:test combineBond
====================*/
TEST_F(DenseUniTensorTest, combineBond) {
  std::vector<std::string> labels = {"a", "b", "c"};
  auto ut = UniTensor({Bond(5), Bond(4), Bond(3)}, labels);
  ut.set_rowrank(1);
  int seed = 0;
  random::uniform_(ut, -100.0, 100.0, seed);
  std::vector<std::string> labels_combine = {"b", "c"};
  ut.combineBond(labels_combine);

  // construct answer directly
  labels = {"a", "b"};
  int rowrank = 1;
  auto ans_ut = UniTensor({Bond(5), Bond(12)}, labels, rowrank);
  auto tens = ut.get_block().reshape({5, 12});
  ans_ut.put_block(tens);

  // compare
  EXPECT_TRUE(AreEqUniTensor(ut, ans_ut));
}

/*=====test info=====
describe:test combineBond with diagonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, combineBond_diag) {
  EXPECT_THROW(ut_complex_diag.combineBond(ut_complex_diag.labels()), std::logic_error);
}

/*=====test info=====
describe:test combineBond error
====================*/
TEST_F(DenseUniTensorTest, combineBond_error) {
  std::vector<std::string> labels = {"a", "b", "c"};
  auto ut = UniTensor({Bond(5), Bond(4), Bond(3)}, labels);
  ut.set_rowrank(1);
  int seed = 0;
  random::uniform_(ut, -100.0, 100.0, seed);

  // not exist labels
  std::vector<std::string> labels_combine = {"c", "d"};
  EXPECT_THROW(ut.combineBond(labels_combine), std::logic_error);

  // empty combine's label
  labels_combine = std::vector<std::string>();
  EXPECT_THROW(ut.combineBond(labels_combine), std::logic_error);
}

/*=====test info=====
describe:test combindbonds with uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, combineBond_ut_uninit) {
  std::vector<std::string> labels_combine = {};
  EXPECT_ANY_THROW(ut_uninit.combineBond(labels_combine));
}

TEST_F(DenseUniTensorTest, contract1) {
  ut1.set_labels({"a", "b", "c", "d"});
  ut2.set_labels({"a", "aa", "bb", "cc"});
  UniTensor out = ut1.contract(ut2);
  auto outbk = out.get_block_();
  auto ansbk = contres1.get_block_();
  EXPECT_TRUE(AreNearlyEqTensor(outbk, ansbk, 1e-5));
}

TEST_F(DenseUniTensorTest, contract2) {
  ut1.set_labels({"a", "b", "c", "d"});
  ut2.set_labels({"a", "b", "bb", "cc"});
  UniTensor out = ut1.contract(ut2);
  auto outbk = out.get_block_();
  auto ansbk = contres2.get_block_();
  EXPECT_TRUE(AreNearlyEqTensor(outbk, ansbk, 1e-5));
}

TEST_F(DenseUniTensorTest, contract3) {
  ut1.set_labels({"a", "b", "c", "d"});
  ut2.set_labels({"a", "b", "c", "cc"});
  UniTensor out = ut1.contract(ut2);
  auto outbk = out.get_block_();
  auto ansbk = contres3.get_block_();
  EXPECT_TRUE(AreNearlyEqTensor(outbk, ansbk, 1e-5));
}

TEST_F(DenseUniTensorTest, same_data) {
  UniTensor B = ut1.permute({1, 0, 3, 2});
  UniTensor C = B.contiguous();
  EXPECT_FALSE(B.same_data(C));
  EXPECT_TRUE(ut1.same_data(B));
}

/*=====test info=====
describe:test add two UniTensor for all data type
====================*/
TEST_F(DenseUniTensorTest, Add_UT_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    UniTensor ut1 = UniTensor(bonds);
    UniTensor ut2 = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut1, 0, 5.0, seed);
      random::uniform_(ut2, 0, 5.0, seed = 1);
      ut1 = ut1.astype(dtype);
      ut2 = ut2.astype(dtype);
    } else {
      ut1 = ut1.astype(dtype);
      ut2 = ut2.astype(dtype);
      random::uniform_(ut1, 0, 5.0, seed);
      random::uniform_(ut2, 0, 5.0, seed = 1);
    }
    auto clone = ut1.clone();
    auto shape = ut1.shape();
    auto out = ut1.Add(ut2);
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          EXPECT_EQ(ut1.at({i, j, k}) + ut2.at({i, j, k}), out.at({i, j, k}));
          EXPECT_EQ(ut1.at({i, j, k}), clone.at({i, j, k}));  // check source not change
        }
      }
    }
  }
}

/*=====test info=====
describe:test add two UniTensor, the second UniTensor only one element
input:
  UT1:A UniTensor with shape [3, 4, 2]
  UT2:A UniTensor with shape [1], only one element
result: return ut1[i] + ut2[0]
====================*/
TEST_F(DenseUniTensorTest, Add_UT_UT1) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  UniTensor ut1 = UniTensor(bonds);
  UniTensor ut2 = UniTensor({Bond(1)});
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  auto out = ut1.Add(ut2);
  for (size_t i = 0; i < shape[0]; i++) {
    for (size_t j = 0; j < shape[1]; j++) {
      for (size_t k = 0; k < shape[2]; k++) {
        EXPECT_EQ(ut1.at({i, j, k}) + ut2.at({0}), out.at({i, j, k}));
        EXPECT_EQ(ut1.at({i, j, k}), clone.at({i, j, k}));  // check source not change
      }
    }
  }
}

/*=====test info=====
describe:test add two UniTensor, the first UniTensor only one element
input:
  UT1:A UniTensor with shape [1], only one element
  UT2:A UniTensor with shape [3, 4, 2]
result: throw error
====================*/
TEST_F(DenseUniTensorTest, Add_UT1_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  UniTensor ut1 = UniTensor({Bond(1)});
  UniTensor ut2 = UniTensor(bonds);
  EXPECT_THROW(ut1.Add(ut2), std::logic_error);
}

/*=====test info=====
describe:test add two UniTensor, both are only 1 element
====================*/
TEST_F(DenseUniTensorTest, Add_UT1_UT1) {
  int seed = 0;
  UniTensor ut1 = UniTensor({Bond(1)});
  UniTensor ut2 = UniTensor({Bond(1)});
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  auto out = ut1.Add(ut2);
  EXPECT_EQ(ut1.at({0}) + ut2.at({0}), out.at({0}));
  EXPECT_EQ(ut1.at({0}), clone.at({0}));  // check source not change
}

/*=====test info=====
describe:test add two UniTensor, one is digonal and the onther is not
====================*/
#if FAIL_CASE_OPEN
TEST_F(DenseUniTensorTest, Add_diag_ndiag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut1 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  is_diag = false;
  auto ut2 = UniTensor(bonds, {"1"}, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  auto out = ut1.Add(ut2);
  auto ut1_nondiag = ut1.to_dense();
  auto ans = ut1_nondiag.Add(ut2);
  EXPECT_TRUE(AreEqUniTensor(out, ans));
}
#endif

/*=====test info=====
describe:test add two diagonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, Add_diag_diag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut1 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  auto ut2 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  auto out = ut1.Add(ut2);
  for (cytnx_uint64 i = 0; i < shape[0]; i++) {
    EXPECT_EQ(ut1.at({i}) + ut2.at({i}), out.at({i}));
    EXPECT_EQ(ut1.at({i}), clone.at({i}));
  }
}

/*=====test info=====
describe:test adding two UniTensor with different shape but not contain 1-element UniTesnor.
====================*/
TEST_F(DenseUniTensorTest, Add_UT_UT_rank_error) {
  auto ut1 = UniTensor({Bond(1), Bond(2)});
  auto ut2 = UniTensor({Bond(1), Bond(3)});
  EXPECT_THROW(ut1.Add(ut2), std::logic_error);
}

/*=====test info=====
describe:test add one UniTensor and one scalar
====================*/
TEST_F(DenseUniTensorTest, Add_UT_Scalar) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    auto cnst = dtype <= Type.ComplexFloat ? Scalar(std::complex<double>(9.1, 2.3)) : Scalar(9);
    cnst = cnst.astype(dtype);
    UniTensor ut = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut, 0, 5.0, seed);
      ut = ut.astype(dtype);
    } else {
      ut = ut.astype(dtype);
      random::uniform_(ut, 0, 5.0, seed);
    }
    auto clone = ut.clone();
    auto shape = ut.shape();
    auto out = ut.Add(cnst);
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          EXPECT_EQ(ut.at({i, j, k}) + cnst, out.at({i, j, k}));
          EXPECT_EQ(ut.at({i, j, k}), clone.at({i, j, k}));  // check source not change
        }
      }
    }
  }
}

/*=====test info=====
describe:test add one diagonal UniTensor and one scalar
====================*/
TEST_F(DenseUniTensorTest, Add_diagUT_Scalar) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut, 0, 5.0, seed);
  auto cnst = Scalar(int(9));
  auto clone = ut.clone();
  auto shape = ut.shape();
  auto out = ut.Add(cnst);
  EXPECT_TRUE(out.is_diag());
  for (size_t i = 0; i < shape[0]; i++) {
    EXPECT_EQ(ut.at({i}) + double(cnst), out.at({i}));
    EXPECT_EQ(ut.at({i}), clone.at({i}));  // check source not change
  }
}

/*=====test info=====
describe:test Add_, two UniTensor for all data type
====================*/
TEST_F(DenseUniTensorTest, Add__UT_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    UniTensor ut1 = UniTensor(bonds);
    UniTensor ut2 = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut1, 0, 5.0, seed);
      random::uniform_(ut2, 0, 5.0, seed = 1);
      ut1 = ut1.astype(dtype);
      ut2 = ut2.astype(dtype);
    } else {
      ut1 = ut1.astype(dtype);
      ut2 = ut2.astype(dtype);
      random::uniform_(ut1, 0, 5.0, seed);
      random::uniform_(ut2, 0, 5.0, seed = 1);
    }
    auto clone = ut1.clone();
    auto shape = ut1.shape();
    ut1.Add_(ut2);
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          EXPECT_EQ(clone.at({i, j, k}) + ut2.at({i, j, k}), ut1.at({i, j, k}));
        }
      }
    }
  }
}

/*=====test info=====
describe:test Add_ two UniTensor, the second UniTensor only one element
input:
  UT1:A UniTensor with shape [3, 4, 2]
  UT2:A UniTensor with shape [1], only one element
result: return ut1[i] = ut1[i] + ut2[0]
====================*/
TEST_F(DenseUniTensorTest, Add__UT_UT1) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  UniTensor ut1 = UniTensor(bonds);
  UniTensor ut2 = UniTensor({Bond(1)});
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  ut1.Add_(ut2);
  for (size_t i = 0; i < shape[0]; i++) {
    for (size_t j = 0; j < shape[1]; j++) {
      for (size_t k = 0; k < shape[2]; k++) {
        EXPECT_EQ(clone.at({i, j, k}) + ut2.at({0}), ut1.at({i, j, k}));
      }
    }
  }
}

/*=====test info=====
describe:test add two UniTensor, the first UniTensor only one element
input:
  UT1:A UniTensor with shape [1], only one element
  UT2:A UniTensor with shape [3, 4, 2]
result: throw error
====================*/
TEST_F(DenseUniTensorTest, Add__UT1_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  UniTensor ut1 = UniTensor({Bond(1)});
  UniTensor ut2 = UniTensor(bonds);
  EXPECT_THROW(ut1.Add_(ut2), std::logic_error);
}

/*=====test info=====
describe:test add two UniTensor, both are only 1 element
====================*/
TEST_F(DenseUniTensorTest, Add__UT1_UT1) {
  int seed = 0;
  UniTensor ut1 = UniTensor({Bond(1)});
  UniTensor ut2 = UniTensor({Bond(1)});
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  ut1.Add_(ut2);
  EXPECT_EQ(clone.at({0}) + ut2.at({0}), ut1.at({0}));
}

/*=====test info=====
describe:test Add_ two UniTensor, one is digonal and the onther is not
====================*/
#if FAIL_CASE_OPEN
TEST_F(DenseUniTensorTest, Add__diag_ndiag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut1 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  is_diag = false;
  auto ut2 = UniTensor(bonds, {"1"}, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto ut1_nondiag = ut1.to_dense();
  ut1.Add_(ut2);
  auto ans = ut1_nondiag.Add(ut2);
  EXPECT_TRUE(AreEqUniTensor(ut1, ans));
}
#endif

/*=====test info=====
describe:test Add_ two diagonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, Add__diag_diag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut1 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  auto ut2 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  ut1.Add_(ut2);
  for (cytnx_uint64 i = 0; i < shape[0]; i++) {
    EXPECT_EQ(clone.at({i}) + ut2.at({i}), ut1.at({i}));
  }
}

/*=====test info=====
describe:test Add_ self
====================*/
TEST_F(DenseUniTensorTest, Add__self) {
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  auto ut = UniTensor(bonds);
  int seed = 0;
  random::uniform_(ut, 0, 5.0, seed);
  auto clone = ut.clone();
  auto shape = ut.shape();
  ut.Add_(ut);
  for (cytnx_uint64 i = 0; i < shape[0]; i++) {
    for (cytnx_uint64 j = 0; j < shape[1]; j++) {
      EXPECT_EQ(clone.at({i, j}) + clone.at({i, j}), ut.at({i, j}));
    }
  }
}

/*=====test info=====
describe:test adding two UniTensor with different shape but not contain 1-element UniTesnor.
====================*/
TEST_F(DenseUniTensorTest, Add__UT_UT_rank_error) {
  auto ut1 = UniTensor({Bond(1), Bond(2)});
  auto ut2 = UniTensor({Bond(1), Bond(3)});
  EXPECT_THROW(ut1.Add_(ut2), std::logic_error);
}

/*=====test info=====
describe:test add one UniTensor and one scalar
====================*/
TEST_F(DenseUniTensorTest, Add__UT_Scalar) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    auto cnst = dtype <= Type.ComplexFloat ? Scalar(std::complex<double>(9.1, 2.3)) : Scalar(9);
    cnst = cnst.astype(dtype);
    UniTensor ut = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut, 0, 5.0, seed);
      ut = ut.astype(dtype);
    } else {
      ut = ut.astype(dtype);
      random::uniform_(ut, 0, 5.0, seed);
    }
    auto clone = ut.clone();
    auto shape = ut.shape();
    ut.Add_(cnst);
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          EXPECT_EQ(clone.at({i, j, k}) + cnst, ut.at({i, j, k}));
        }
      }
    }
  }
}

/*=====test info=====
describe:test add one diagonal UniTensor and one scalar
====================*/
TEST_F(DenseUniTensorTest, Add__diagUT_Scalar) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut, 0, 5.0, seed);
  auto cnst = Scalar(int(9));
  auto clone = ut.clone();
  auto shape = ut.shape();
  ut.Add_(cnst);
  EXPECT_TRUE(ut.is_diag());
  for (size_t i = 0; i < shape[0]; i++) {
    EXPECT_EQ(clone.at({i}) + double(cnst), ut.at({i}));
  }
}

/*=====test info=====
describe:test oprator+=, input two UniTensor
====================*/
TEST_F(DenseUniTensorTest, operatorAdd_UT_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  UniTensor ut1 = UniTensor(bonds);
  UniTensor ut2 = UniTensor(bonds);
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto ans = ut1.clone();
  ans.Add_(ut2);
  ut1 += ut2;
  EXPECT_TRUE(AreEqUniTensor(ut1, ans));
}

/*=====test info=====
describe:test oprator+=, input one UniTensor and one Scalar
====================*/
TEST_F(DenseUniTensorTest, operatorAdd_UT_Scalar) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  UniTensor ut = UniTensor(bonds);
  random::uniform_(ut, 0, 5.0, seed);
  auto cnst = Scalar(9.2);
  auto ans = ut.clone();
  ans.Add_(cnst);
  ut += cnst;
  EXPECT_TRUE(AreEqUniTensor(ut, ans));
}

/*=====test info=====
describe:test Add related function with uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, Add_uninit) {
  EXPECT_ANY_THROW(utzero345.Add(ut_uninit));
  EXPECT_ANY_THROW(utzero345.Add_(ut_uninit));
  EXPECT_ANY_THROW(ut_uninit.Add(utzero345));
  EXPECT_ANY_THROW(ut_uninit.Add_(utzero345));
  EXPECT_ANY_THROW(ut_uninit.Add_(ut_uninit));

  auto cnst = Scalar(9.2);
  EXPECT_ANY_THROW(ut_uninit.Add(cnst));
  EXPECT_ANY_THROW(ut_uninit.Add_(cnst));

  EXPECT_ANY_THROW(ut_uninit += utzero345);
  EXPECT_ANY_THROW(ut_uninit += cnst);
}

/*=====test info=====
describe:test sub two UniTensor for all data type
====================*/
TEST_F(DenseUniTensorTest, Sub_UT_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    UniTensor ut1 = UniTensor(bonds);
    UniTensor ut2 = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut1, 0, 5.0, seed);
      random::uniform_(ut2, 0, 5.0, seed = 1);
      ut1 = ut1.astype(dtype);
      ut2 = ut2.astype(dtype);
    } else {
      ut1 = ut1.astype(dtype);
      ut2 = ut2.astype(dtype);
      random::uniform_(ut1, 0, 5.0, seed);
      random::uniform_(ut2, 0, 5.0, seed = 1);
    }
    auto clone = ut1.clone();
    auto shape = ut1.shape();
    auto out = ut1.Sub(ut2);
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          // issue: for dtype = Uint64, the overload of operator- may not correct => 2 - 2 = 4.
          EXPECT_EQ(ut1.at({i, j, k}) - ut2.at({i, j, k}), out.at({i, j, k}));
          EXPECT_EQ(ut1.at({i, j, k}), clone.at({i, j, k}));  // check source not change
        }
      }
    }
  }
}

/*=====test info=====
describe:test sub two UniTensor, the second UniTensor only one element
input:
  UT1:A UniTensor with shape [3, 4, 2]
  UT2:A UniTensor with shape [1], only one element
result: return ut1[i] + ut2[0]
====================*/
TEST_F(DenseUniTensorTest, Sub_UT_UT1) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  UniTensor ut1 = UniTensor(bonds);
  UniTensor ut2 = UniTensor({Bond(1)});
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  auto out = ut1.Sub(ut2);
  for (size_t i = 0; i < shape[0]; i++) {
    for (size_t j = 0; j < shape[1]; j++) {
      for (size_t k = 0; k < shape[2]; k++) {
        EXPECT_EQ(ut1.at({i, j, k}) - ut2.at({0}), out.at({i, j, k}));
        EXPECT_EQ(ut1.at({i, j, k}), clone.at({i, j, k}));  // check source not change
      }
    }
  }
}

/*=====test info=====
describe:test sub two UniTensor, the first UniTensor only one element
input:
  UT1:A UniTensor with shape [1], only one element
  UT2:A UniTensor with shape [3, 4, 2]
result: throw error
====================*/
TEST_F(DenseUniTensorTest, Sub_UT1_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  UniTensor ut1 = UniTensor({Bond(1)});
  UniTensor ut2 = UniTensor(bonds);
  EXPECT_THROW(ut1.Sub(ut2), std::logic_error);
}

/*=====test info=====
describe:test sub two UniTensor, both are only 1 element
====================*/
TEST_F(DenseUniTensorTest, Sub_UT1_UT1) {
  int seed = 0;
  UniTensor ut1 = UniTensor({Bond(1)});
  UniTensor ut2 = UniTensor({Bond(1)});
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  auto out = ut1.Sub(ut2);
  EXPECT_EQ(ut1.at({0}) - ut2.at({0}), out.at({0}));
  EXPECT_EQ(ut1.at({0}), clone.at({0}));  // check source not change
}

/*=====test info=====
describe:test sub two UniTensor, one is digonal and the onther is not
====================*/
#if FAIL_CASE_OPEN
TEST_F(DenseUniTensorTest, Sub_diag_ndiag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut1 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  is_diag = false;
  auto ut2 = UniTensor(bonds, {"1"}, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  auto out = ut1.Sub(ut2);
  auto ut1_nondiag = ut1.to_dense();
  auto ans = ut1_nondiag.Sub(ut2);
  EXPECT_TRUE(AreEqUniTensor(out, ans));
}
#endif

/*=====test info=====
describe:test sub two diagonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, Sub_diag_diag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut1 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  auto ut2 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  auto out = ut1.Sub(ut2);
  for (cytnx_uint64 i = 0; i < shape[0]; i++) {
    EXPECT_EQ(ut1.at({i}) - ut2.at({i}), out.at({i}));
    EXPECT_EQ(ut1.at({i}), clone.at({i}));
  }
}

/*=====test info=====
describe:test subing two UniTensor with different shape but not contain 1-element UniTesnor.
====================*/
TEST_F(DenseUniTensorTest, Sub_UT_UT_rank_error) {
  auto ut1 = UniTensor({Bond(1), Bond(2)});
  auto ut2 = UniTensor({Bond(1), Bond(3)});
  EXPECT_THROW(ut1.Sub(ut2), std::logic_error);
}

/*=====test info=====
describe:test sub one UniTensor and one scalar
====================*/
TEST_F(DenseUniTensorTest, Sub_UT_Scalar) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    auto cnst = dtype <= Type.ComplexFloat ? Scalar(std::complex<double>(9.1, 2.3)) : Scalar(9);
    cnst = cnst.astype(dtype);
    UniTensor ut = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut, 0, 5.0, seed);
      ut = ut.astype(dtype);
    } else {
      ut = ut.astype(dtype);
      random::uniform_(ut, 0, 5.0, seed);
    }
    auto clone = ut.clone();
    auto shape = ut.shape();
    auto out = ut.Sub(cnst);
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          EXPECT_EQ(ut.at({i, j, k}) - cnst, out.at({i, j, k}));
          EXPECT_EQ(ut.at({i, j, k}), clone.at({i, j, k}));  // check source not change
        }
      }
    }
  }
}

/*=====test info=====
describe:test sub one diagonal UniTensor and one scalar
====================*/
TEST_F(DenseUniTensorTest, Sub_diagUT_Scalar) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut, 0, 5.0, seed);
  auto cnst = Scalar(int(9));
  auto clone = ut.clone();
  auto shape = ut.shape();
  auto out = ut.Sub(cnst);
  EXPECT_TRUE(out.is_diag());
  for (size_t i = 0; i < shape[0]; i++) {
    EXPECT_EQ(ut.at({i}) - double(cnst), out.at({i}));
    EXPECT_EQ(ut.at({i}), clone.at({i}));  // check source not change
  }
}

/*=====test info=====
describe:test Sub_, two UniTensor for all data type
====================*/
TEST_F(DenseUniTensorTest, Sub__UT_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    UniTensor ut1 = UniTensor(bonds);
    UniTensor ut2 = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut1, 0, 5.0, seed);
      random::uniform_(ut2, 0, 5.0, seed = 1);
      ut1 = ut1.astype(dtype);
      ut2 = ut2.astype(dtype);
    } else {
      ut1 = ut1.astype(dtype);
      ut2 = ut2.astype(dtype);
      random::uniform_(ut1, 0, 5.0, seed);
      random::uniform_(ut2, 0, 5.0, seed = 1);
    }
    auto clone = ut1.clone();
    auto shape = ut1.shape();
    ut1.Sub_(ut2);
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          EXPECT_EQ(clone.at({i, j, k}) - ut2.at({i, j, k}), ut1.at({i, j, k}));
        }
      }
    }
  }
}

/*=====test info=====
describe:test Sub_ two UniTensor, the second UniTensor only one element
input:
  UT1:A UniTensor with shape [3, 4, 2]
  UT2:A UniTensor with shape [1], only one element
result: return ut1[i] = ut1[i] + ut2[0]
====================*/
TEST_F(DenseUniTensorTest, Sub__UT_UT1) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  UniTensor ut1 = UniTensor(bonds);
  UniTensor ut2 = UniTensor({Bond(1)});
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  ut1.Sub_(ut2);
  for (size_t i = 0; i < shape[0]; i++) {
    for (size_t j = 0; j < shape[1]; j++) {
      for (size_t k = 0; k < shape[2]; k++) {
        EXPECT_EQ(clone.at({i, j, k}) - ut2.at({0}), ut1.at({i, j, k}));
      }
    }
  }
}

/*=====test info=====
describe:test sub two UniTensor, the first UniTensor only one element
input:
  UT1:A UniTensor with shape [1], only one element
  UT2:A UniTensor with shape [3, 4, 2]
result: throw error
====================*/
TEST_F(DenseUniTensorTest, Sub__UT1_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  UniTensor ut1 = UniTensor({Bond(1)});
  UniTensor ut2 = UniTensor(bonds);
  EXPECT_THROW(ut1.Sub_(ut2), std::logic_error);
}

/*=====test info=====
describe:test sub two UniTensor, both are only 1 element
====================*/
TEST_F(DenseUniTensorTest, Sub__UT1_UT1) {
  int seed = 0;
  UniTensor ut1 = UniTensor({Bond(1)});
  UniTensor ut2 = UniTensor({Bond(1)});
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  ut1.Sub_(ut2);
  EXPECT_EQ(clone.at({0}) - ut2.at({0}), ut1.at({0}));
}

/*=====test info=====
describe:test Sub_ two UniTensor, one is digonal and the onther is not
====================*/
#if FAIL_CASE_OPEN
TEST_F(DenseUniTensorTest, Sub__diag_ndiag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut1 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  is_diag = false;
  auto ut2 = UniTensor(bonds, {"1"}, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto ut1_nondiag = ut1.to_dense();
  ut1.Sub_(ut2);
  auto ans = ut1_nondiag.Sub(ut2);
  EXPECT_TRUE(AreEqUniTensor(ut1, ans));
}
#endif

/*=====test info=====
describe:test Sub_ two diagonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, Sub__diag_diag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut1 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  auto ut2 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  ut1.Sub_(ut2);
  for (cytnx_uint64 i = 0; i < shape[0]; i++) {
    EXPECT_EQ(clone.at({i}) - ut2.at({i}), ut1.at({i}));
  }
}

/*=====test info=====
describe:test Sub_ self
====================*/
TEST_F(DenseUniTensorTest, Sub__self) {
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  auto ut = UniTensor(bonds);
  int seed = 0;
  random::uniform_(ut, 0, 5.0, seed);
  auto clone = ut.clone();
  auto shape = ut.shape();
  ut.Sub_(ut);
  for (cytnx_uint64 i = 0; i < shape[0]; i++) {
    for (cytnx_uint64 j = 0; j < shape[1]; j++) {
      EXPECT_EQ(clone.at({i, j}) - clone.at({i, j}), ut.at({i, j}));
    }
  }
}

/*=====test info=====
describe:test subing two UniTensor with different shape but not contain 1-element UniTesnor.
====================*/
TEST_F(DenseUniTensorTest, Sub__UT_UT_rank_error) {
  auto ut1 = UniTensor({Bond(1), Bond(2)});
  auto ut2 = UniTensor({Bond(1), Bond(3)});
  EXPECT_THROW(ut1.Sub_(ut2), std::logic_error);
}

/*=====test info=====
describe:test sub one UniTensor and one scalar
====================*/
TEST_F(DenseUniTensorTest, Sub__UT_Scalar) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    auto cnst = dtype <= Type.ComplexFloat ? Scalar(std::complex<double>(9.1, 2.3)) : Scalar(9);
    cnst = cnst.astype(dtype);
    UniTensor ut = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut, 0, 5.0, seed);
      ut = ut.astype(dtype);
    } else {
      ut = ut.astype(dtype);
      random::uniform_(ut, 0, 5.0, seed);
    }
    auto clone = ut.clone();
    auto shape = ut.shape();
    ut.Sub_(cnst);
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          EXPECT_EQ(clone.at({i, j, k}) - cnst, ut.at({i, j, k}));
        }
      }
    }
  }
}

/*=====test info=====
describe:test sub one diagonal UniTensor and one scalar
====================*/
TEST_F(DenseUniTensorTest, Sub__diagUT_Scalar) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut, 0, 5.0, seed);
  auto cnst = Scalar(int(9));
  auto clone = ut.clone();
  auto shape = ut.shape();
  ut.Sub_(cnst);
  EXPECT_TRUE(ut.is_diag());
  for (size_t i = 0; i < shape[0]; i++) {
    EXPECT_EQ(clone.at({i}) - double(cnst), ut.at({i}));
  }
}

/*=====test info=====
describe:test oprator+=, input two UniTensor
====================*/
TEST_F(DenseUniTensorTest, operatorSub_UT_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  UniTensor ut1 = UniTensor(bonds);
  UniTensor ut2 = UniTensor(bonds);
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto ans = ut1.clone();
  ans.Sub_(ut2);
  ut1 -= ut2;
  EXPECT_TRUE(AreEqUniTensor(ut1, ans));
}

/*=====test info=====
describe:test oprator+=, input one UniTensor and one Scalar
====================*/
TEST_F(DenseUniTensorTest, operatorSub_UT_Scalar) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  UniTensor ut = UniTensor(bonds);
  random::uniform_(ut, 0, 5.0, seed);
  auto cnst = Scalar(9.2);
  auto ans = ut.clone();
  ans.Sub_(cnst);
  ut -= cnst;
  EXPECT_TRUE(AreEqUniTensor(ut, ans));
}

/*=====test info=====
describe:test Sub related function with uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, Sub_uninit) {
  EXPECT_ANY_THROW(utzero345.Sub(ut_uninit));
  EXPECT_ANY_THROW(utzero345.Sub_(ut_uninit));
  EXPECT_ANY_THROW(ut_uninit.Sub(utzero345));
  EXPECT_ANY_THROW(ut_uninit.Sub_(utzero345));
  EXPECT_ANY_THROW(ut_uninit.Sub_(ut_uninit));

  auto cnst = Scalar(9.2);
  EXPECT_ANY_THROW(ut_uninit.Sub(cnst));
  EXPECT_ANY_THROW(ut_uninit.Sub_(cnst));

  EXPECT_ANY_THROW(ut_uninit -= utzero345);
  EXPECT_ANY_THROW(ut_uninit -= cnst);
}

/*=====test info=====
describe:test mul two UniTensor for all data type
====================*/
TEST_F(DenseUniTensorTest, Mul_UT_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    UniTensor ut1 = UniTensor(bonds);
    UniTensor ut2 = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut1, 0, 5.0, seed);
      random::uniform_(ut2, 0, 5.0, seed = 1);
      ut1 = ut1.astype(dtype);
      ut2 = ut2.astype(dtype);
    } else {
      ut1 = ut1.astype(dtype);
      ut2 = ut2.astype(dtype);
      random::uniform_(ut1, 0, 5.0, seed);
      random::uniform_(ut2, 0, 5.0, seed = 1);
    }
    auto clone = ut1.clone();
    auto shape = ut1.shape();
    auto out = ut1.Mul(ut2);
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          EXPECT_EQ(ut1.at({i, j, k}) * ut2.at({i, j, k}), out.at({i, j, k}));
          EXPECT_EQ(ut1.at({i, j, k}), clone.at({i, j, k}));  // check source not change
        }
      }
    }
  }
}

/*=====test info=====
describe:test mul two UniTensor, the second UniTensor only one element
input:
  UT1:A UniTensor with shape [3, 4, 2]
  UT2:A UniTensor with shape [1], only one element
result: return ut1[i] + ut2[0]
====================*/
TEST_F(DenseUniTensorTest, Mul_UT_UT1) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  UniTensor ut1 = UniTensor(bonds);
  UniTensor ut2 = UniTensor({Bond(1)});
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  auto out = ut1.Mul(ut2);
  for (size_t i = 0; i < shape[0]; i++) {
    for (size_t j = 0; j < shape[1]; j++) {
      for (size_t k = 0; k < shape[2]; k++) {
        EXPECT_EQ(ut1.at({i, j, k}) * ut2.at({0}), out.at({i, j, k}));
        EXPECT_EQ(ut1.at({i, j, k}), clone.at({i, j, k}));  // check source not change
      }
    }
  }
}

/*=====test info=====
describe:test mul two UniTensor, the first UniTensor only one element
input:
  UT1:A UniTensor with shape [1], only one element
  UT2:A UniTensor with shape [3, 4, 2]
result: throw error
====================*/
TEST_F(DenseUniTensorTest, Mul_UT1_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  UniTensor ut1 = UniTensor({Bond(1)});
  UniTensor ut2 = UniTensor(bonds);
  EXPECT_THROW(ut1.Mul(ut2), std::logic_error);
}

/*=====test info=====
describe:test mul two UniTensor, both are only 1 element
====================*/
TEST_F(DenseUniTensorTest, Mul_UT1_UT1) {
  int seed = 0;
  UniTensor ut1 = UniTensor({Bond(1)});
  UniTensor ut2 = UniTensor({Bond(1)});
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  auto out = ut1.Mul(ut2);
  EXPECT_EQ(ut1.at({0}) * ut2.at({0}), out.at({0}));
  EXPECT_EQ(ut1.at({0}), clone.at({0}));  // check source not change
}

/*=====test info=====
describe:test mul two UniTensor, one is digonal and the onther is not
====================*/
#if FAIL_CASE_OPEN
TEST_F(DenseUniTensorTest, Mul_diag_ndiag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut1 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  is_diag = false;
  auto ut2 = UniTensor(bonds, {"1"}, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  auto out = ut1.Mul(ut2);
  auto ut1_nondiag = ut1.to_dense();
  auto ans = ut1_nondiag.Mul(ut2);
  EXPECT_TRUE(AreEqUniTensor(out, ans));
}
#endif

/*=====test info=====
describe:test mul two diagonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, Mul_diag_diag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut1 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  auto ut2 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  auto out = ut1.Mul(ut2);
  for (cytnx_uint64 i = 0; i < shape[0]; i++) {
    EXPECT_EQ(ut1.at({i}) * ut2.at({i}), out.at({i}));
    EXPECT_EQ(ut1.at({i}), clone.at({i}));
  }
}

/*=====test info=====
describe:test muling two UniTensor with different shape but not contain 1-element UniTesnor.
====================*/
TEST_F(DenseUniTensorTest, Mul_UT_UT_rank_error) {
  auto ut1 = UniTensor({Bond(1), Bond(2)});
  auto ut2 = UniTensor({Bond(1), Bond(3)});
  EXPECT_THROW(ut1.Mul(ut2), std::logic_error);
}

/*=====test info=====
describe:test mul one UniTensor and one scalar
====================*/
TEST_F(DenseUniTensorTest, Mul_UT_Scalar) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    auto cnst = dtype <= Type.ComplexFloat ? Scalar(std::complex<double>(9.1, 2.3)) : Scalar(9);
    cnst = cnst.astype(dtype);
    UniTensor ut = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut, 0, 5.0, seed);
      ut = ut.astype(dtype);
    } else {
      ut = ut.astype(dtype);
      random::uniform_(ut, 0, 5.0, seed);
    }
    auto clone = ut.clone();
    auto shape = ut.shape();
    auto out = ut.Mul(cnst);
    const double tol = 1.0e-5;
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          auto diff = abs(ut.at({i, j, k}) * cnst - out.at({i, j, k}));
          EXPECT_TRUE(diff <= tol);
        }
      }
    }
  }
}

/*=====test info=====
describe:test mul one diagonal UniTensor and one scalar
====================*/
TEST_F(DenseUniTensorTest, Mul_diagUT_Scalar) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut, 0, 5.0, seed);
  auto cnst = Scalar(int(9));
  auto clone = ut.clone();
  auto shape = ut.shape();
  auto out = ut.Mul(cnst);
  EXPECT_TRUE(out.is_diag());
  for (size_t i = 0; i < shape[0]; i++) {
    EXPECT_EQ(ut.at({i}) * double(cnst), out.at({i}));
    EXPECT_EQ(ut.at({i}), clone.at({i}));  // check source not change
  }
}

/*=====test info=====
describe:test Mul_, two UniTensor for all data type
====================*/
TEST_F(DenseUniTensorTest, Mul__UT_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    UniTensor ut1 = UniTensor(bonds);
    UniTensor ut2 = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut1, 0, 5.0, seed);
      random::uniform_(ut2, 0, 5.0, seed = 1);
      ut1 = ut1.astype(dtype);
      ut2 = ut2.astype(dtype);
    } else {
      ut1 = ut1.astype(dtype);
      ut2 = ut2.astype(dtype);
      random::uniform_(ut1, 0, 5.0, seed);
      random::uniform_(ut2, 0, 5.0, seed = 1);
    }
    auto clone = ut1.clone();
    auto shape = ut1.shape();
    ut1.Mul_(ut2);
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          EXPECT_EQ(clone.at({i, j, k}) * ut2.at({i, j, k}), ut1.at({i, j, k}));
        }
      }
    }
  }
}

/*=====test info=====
describe:test Mul_ two UniTensor, the second UniTensor only one element
input:
  UT1:A UniTensor with shape [3, 4, 2]
  UT2:A UniTensor with shape [1], only one element
result: return ut1[i] = ut1[i] + ut2[0]
====================*/
TEST_F(DenseUniTensorTest, Mul__UT_UT1) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  UniTensor ut1 = UniTensor(bonds);
  UniTensor ut2 = UniTensor({Bond(1)});
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  ut1.Mul_(ut2);
  for (size_t i = 0; i < shape[0]; i++) {
    for (size_t j = 0; j < shape[1]; j++) {
      for (size_t k = 0; k < shape[2]; k++) {
        EXPECT_EQ(clone.at({i, j, k}) * ut2.at({0}), ut1.at({i, j, k}));
      }
    }
  }
}

/*=====test info=====
describe:test mul two UniTensor, the first UniTensor only one element
input:
  UT1:A UniTensor with shape [1], only one element
  UT2:A UniTensor with shape [3, 4, 2]
result: throw error
====================*/
TEST_F(DenseUniTensorTest, Mul__UT1_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  UniTensor ut1 = UniTensor({Bond(1)});
  UniTensor ut2 = UniTensor(bonds);
  EXPECT_THROW(ut1.Mul_(ut2), std::logic_error);
}

/*=====test info=====
describe:test mul two UniTensor, both are only 1 element
====================*/
TEST_F(DenseUniTensorTest, Mul__UT1_UT1) {
  int seed = 0;
  UniTensor ut1 = UniTensor({Bond(1)});
  UniTensor ut2 = UniTensor({Bond(1)});
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  ut1.Mul_(ut2);
  EXPECT_EQ(clone.at({0}) * ut2.at({0}), ut1.at({0}));
}

/*=====test info=====
describe:test Mul_ two UniTensor, one is digonal and the onther is not
====================*/
#if FAIL_CASE_OPEN
TEST_F(DenseUniTensorTest, Mul__diag_ndiag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut1 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  is_diag = false;
  auto ut2 = UniTensor(bonds, {"1"}, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto ut1_nondiag = ut1.to_dense();
  ut1.Mul_(ut2);
  auto ans = ut1_nondiag.Mul(ut2);
  EXPECT_TRUE(AreEqUniTensor(ut1, ans));
}
#endif

/*=====test info=====
describe:test Mul_ two diagonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, Mul__diag_diag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut1 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  auto ut2 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  ut1.Mul_(ut2);
  for (cytnx_uint64 i = 0; i < shape[0]; i++) {
    EXPECT_EQ(clone.at({i}) * ut2.at({i}), ut1.at({i}));
  }
}

/*=====test info=====
describe:test Mul_ self
====================*/
TEST_F(DenseUniTensorTest, Mul__self) {
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  auto ut = UniTensor(bonds);
  int seed = 0;
  random::uniform_(ut, 0, 5.0, seed);
  auto clone = ut.clone();
  auto shape = ut.shape();
  ut.Mul_(ut);
  for (cytnx_uint64 i = 0; i < shape[0]; i++) {
    for (cytnx_uint64 j = 0; j < shape[1]; j++) {
      EXPECT_EQ(clone.at({i, j}) * clone.at({i, j}), ut.at({i, j}));
    }
  }
}

/*=====test info=====
describe:test muling two UniTensor with different shape but not contain 1-element UniTesnor.
====================*/
TEST_F(DenseUniTensorTest, Mul__UT_UT_rank_error) {
  auto ut1 = UniTensor({Bond(1), Bond(2)});
  auto ut2 = UniTensor({Bond(1), Bond(3)});
  EXPECT_THROW(ut1.Mul_(ut2), std::logic_error);
}

/*=====test info=====
describe:test mul one UniTensor and one scalar
====================*/
TEST_F(DenseUniTensorTest, Mul__UT_Scalar) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    auto cnst = dtype <= Type.ComplexFloat ? Scalar(std::complex<double>(9.1, 2.3)) : Scalar(9);
    cnst = cnst.astype(dtype);
    UniTensor ut = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut, 0, 5.0, seed);
      ut = ut.astype(dtype);
    } else {
      ut = ut.astype(dtype);
      random::uniform_(ut, 0, 5.0, seed);
    }
    auto clone = ut.clone();
    auto shape = ut.shape();
    ut.Mul_(cnst);
    const double tol = 1.0e-5;
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          auto diff = abs(clone.at({i, j, k}) * cnst - ut.at({i, j, k}));
          EXPECT_TRUE(diff <= tol);
        }
      }
    }
  }
}

/*=====test info=====
describe:test mul one diagonal UniTensor and one scalar
====================*/
TEST_F(DenseUniTensorTest, Mul__diagUT_Scalar) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut, 0, 5.0, seed);
  auto cnst = Scalar(int(9));
  auto clone = ut.clone();
  auto shape = ut.shape();
  ut.Mul_(cnst);
  EXPECT_TRUE(ut.is_diag());
  for (size_t i = 0; i < shape[0]; i++) {
    EXPECT_EQ(clone.at({i}) * double(cnst), ut.at({i}));
  }
}

/*=====test info=====
describe:test oprator+=, input two UniTensor
====================*/
TEST_F(DenseUniTensorTest, operatorMul_UT_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  UniTensor ut1 = UniTensor(bonds);
  UniTensor ut2 = UniTensor(bonds);
  random::uniform_(ut1, 0, 5.0, seed);
  random::uniform_(ut2, 0, 5.0, seed = 1);
  auto ans = ut1.clone();
  ans.Mul_(ut2);
  ut1 *= ut2;
  EXPECT_TRUE(AreEqUniTensor(ut1, ans));
}

/*=====test info=====
describe:test oprator+=, input one UniTensor and one Scalar
====================*/
TEST_F(DenseUniTensorTest, operatorMul_UT_Scalar) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  UniTensor ut = UniTensor(bonds);
  random::uniform_(ut, 0, 5.0, seed);
  auto cnst = Scalar(9.2);
  auto ans = ut.clone();
  ans.Mul_(cnst);
  ut *= cnst;
  EXPECT_TRUE(AreEqUniTensor(ut, ans));
}

/*=====test info=====
describe:test Mul related function with uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, Mul_uninit) {
  EXPECT_ANY_THROW(utzero345.Mul(ut_uninit));
  EXPECT_ANY_THROW(utzero345.Mul_(ut_uninit));
  EXPECT_ANY_THROW(ut_uninit.Mul(utzero345));
  EXPECT_ANY_THROW(ut_uninit.Mul_(utzero345));
  EXPECT_ANY_THROW(ut_uninit.Mul_(ut_uninit));

  auto cnst = Scalar(9.2);
  EXPECT_ANY_THROW(ut_uninit.Mul(cnst));
  EXPECT_ANY_THROW(ut_uninit.Mul_(cnst));

  EXPECT_ANY_THROW(ut_uninit *= utzero345);
  EXPECT_ANY_THROW(ut_uninit *= cnst);
}

/*=====test info=====
describe:test div two UniTensor for all data type
====================*/
TEST_F(DenseUniTensorTest, Div_UT_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    UniTensor ut1 = UniTensor(bonds);
    UniTensor ut2 = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut1, 1, 5.0, seed);
      random::uniform_(ut2, 1, 5.0, seed = 1);
      ut1 = ut1.astype(dtype);
      ut2 = ut2.astype(dtype);
    } else {
      ut1 = ut1.astype(dtype);
      ut2 = ut2.astype(dtype);
      random::uniform_(ut1, 1, 5.0, seed);
      random::uniform_(ut2, 1, 5.0, seed = 1);
    }
    auto clone = ut1.clone();
    auto shape = ut1.shape();
    auto out = ut1.Div(ut2);
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          EXPECT_EQ((ut1.at({i, j, k}) / ut2.at({i, j, k})), out.at({i, j, k}));
          EXPECT_EQ(ut1.at({i, j, k}), clone.at({i, j, k}));  // check source not change
        }
      }
    }
  }
}

/*=====test info=====
describe:test div two UniTensor, the second UniTensor only one element
input:
  UT1:A UniTensor with shape [3, 4, 2]
  UT2:A UniTensor with shape [1], only one element
result: return ut1[i] + ut2[0]
====================*/
TEST_F(DenseUniTensorTest, Div_UT_UT1) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  UniTensor ut1 = UniTensor(bonds);
  UniTensor ut2 = UniTensor({Bond(1)});
  random::uniform_(ut1, 1, 5.0, seed);
  random::uniform_(ut2, 1, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  auto out = ut1.Div(ut2);
  for (size_t i = 0; i < shape[0]; i++) {
    for (size_t j = 0; j < shape[1]; j++) {
      for (size_t k = 0; k < shape[2]; k++) {
        EXPECT_EQ(ut1.at({i, j, k}) / ut2.at({0}), out.at({i, j, k}));
        EXPECT_EQ(ut1.at({i, j, k}), clone.at({i, j, k}));  // check source not change
      }
    }
  }
}

/*=====test info=====
describe:test div two UniTensor, the first UniTensor only one element
input:
  UT1:A UniTensor with shape [1], only one element
  UT2:A UniTensor with shape [3, 4, 2]
result: throw error
====================*/
TEST_F(DenseUniTensorTest, Div_UT1_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  UniTensor ut1 = UniTensor({Bond(1)});
  UniTensor ut2 = UniTensor(bonds);
  EXPECT_THROW(ut1.Div(ut2), std::logic_error);
}

/*=====test info=====
describe:test div two UniTensor, both are only 1 element
====================*/
TEST_F(DenseUniTensorTest, Div_UT1_UT1) {
  int seed = 0;
  UniTensor ut1 = UniTensor({Bond(1)});
  UniTensor ut2 = UniTensor({Bond(1)});
  random::uniform_(ut1, 1, 5.0, seed);
  random::uniform_(ut2, 1, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  auto out = ut1.Div(ut2);
  EXPECT_EQ(ut1.at({0}) / ut2.at({0}), out.at({0}));
  EXPECT_EQ(ut1.at({0}), clone.at({0}));  // check source not change
}

/*=====test info=====
describe:test div two UniTensor, one is digonal and the onther is not
====================*/
#if FAIL_CASE_OPEN
TEST_F(DenseUniTensorTest, Div_diag_ndiag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut1 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  is_diag = false;
  auto ut2 = UniTensor(bonds, {"1"}, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut1, 1, 5.0, seed);
  random::uniform_(ut2, 1, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  auto out = ut1.Div(ut2);
  auto ut1_nondiag = ut1.to_dense();
  auto ans = ut1_nondiag.Div(ut2);
  EXPECT_TRUE(AreEqUniTensor(out, ans));
}
#endif

/*=====test info=====
describe:test div two diagonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, Div_diag_diag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut1 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  auto ut2 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut1, 1, 5.0, seed);
  random::uniform_(ut2, 1, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  auto out = ut1.Div(ut2);
  for (cytnx_uint64 i = 0; i < shape[0]; i++) {
    EXPECT_EQ(ut1.at({i}) / ut2.at({i}), out.at({i}));
    EXPECT_EQ(ut1.at({i}), clone.at({i}));
  }
}

/*=====test info=====
describe:test diving two UniTensor with different shape but not contain 1-element UniTesnor.
====================*/
TEST_F(DenseUniTensorTest, Div_UT_UT_rank_error) {
  auto ut1 = UniTensor({Bond(1), Bond(2)});
  auto ut2 = UniTensor({Bond(1), Bond(3)});
  EXPECT_THROW(ut1.Div(ut2), std::logic_error);
}

/*=====test info=====
describe:test div one UniTensor and one scalar
====================*/
TEST_F(DenseUniTensorTest, Div_UT_Scalar) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    auto cnst = dtype <= Type.ComplexFloat ? Scalar(std::complex<double>(9.1, 2.3)) : Scalar(9);
    cnst = cnst.astype(dtype);
    UniTensor ut = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut, 1, 5.0, seed);
      ut = ut.astype(dtype);
    } else {
      ut = ut.astype(dtype);
      random::uniform_(ut, 1, 5.0, seed);
    }
    auto clone = ut.clone();
    auto shape = ut.shape();
    auto out = ut.Div(cnst);
    const double tol = 1.0e-5;
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          auto diff = abs(ut.at({i, j, k}) / cnst - out.at({i, j, k}));
          EXPECT_TRUE(diff <= tol);
        }
      }
    }
  }
}

/*=====test info=====
describe:test div one diagonal UniTensor and one scalar
====================*/
TEST_F(DenseUniTensorTest, Div_diagUT_Scalar) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut, 0, 5.0, seed);
  auto cnst = Scalar(int(9));
  auto clone = ut.clone();
  auto shape = ut.shape();
  auto out = ut.Div(cnst);
  EXPECT_TRUE(out.is_diag());
  for (size_t i = 0; i < shape[0]; i++) {
    EXPECT_DOUBLE_EQ(ut.at<double>({i}) / double(cnst), out.at<double>({i}));
    EXPECT_EQ(ut.at({i}), clone.at({i}));  // check source not change
  }
}

/*=====test info=====
describe:test Div_, two UniTensor for all data type
====================*/
TEST_F(DenseUniTensorTest, Div__UT_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    UniTensor ut1 = UniTensor(bonds);
    UniTensor ut2 = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut1, 1, 5.0, seed);
      random::uniform_(ut2, 1, 5.0, seed = 1);
      ut1 = ut1.astype(dtype);
      ut2 = ut2.astype(dtype);
    } else {
      ut1 = ut1.astype(dtype);
      ut2 = ut2.astype(dtype);
      random::uniform_(ut1, 1, 5.0, seed);
      random::uniform_(ut2, 1, 5.0, seed = 1);
    }
    auto clone = ut1.clone();
    auto shape = ut1.shape();
    ut1.Div_(ut2);
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          EXPECT_EQ(clone.at({i, j, k}) / ut2.at({i, j, k}), ut1.at({i, j, k}));
        }
      }
    }
  }
}

/*=====test info=====
describe:test Div_ two UniTensor, the second UniTensor only one element
input:
  UT1:A UniTensor with shape [3, 4, 2]
  UT2:A UniTensor with shape [1], only one element
result: return ut1[i] = ut1[i] + ut2[0]
====================*/
TEST_F(DenseUniTensorTest, Div__UT_UT1) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  UniTensor ut1 = UniTensor(bonds);
  UniTensor ut2 = UniTensor({Bond(1)});
  random::uniform_(ut1, 1, 5.0, seed);
  random::uniform_(ut2, 1, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  ut1.Div_(ut2);
  for (size_t i = 0; i < shape[0]; i++) {
    for (size_t j = 0; j < shape[1]; j++) {
      for (size_t k = 0; k < shape[2]; k++) {
        EXPECT_EQ(clone.at({i, j, k}) / ut2.at({0}), ut1.at({i, j, k}));
      }
    }
  }
}

/*=====test info=====
describe:test div two UniTensor, the first UniTensor only one element
input:
  UT1:A UniTensor with shape [1], only one element
  UT2:A UniTensor with shape [3, 4, 2]
result: throw error
====================*/
TEST_F(DenseUniTensorTest, Div__UT1_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  UniTensor ut1 = UniTensor({Bond(1)});
  UniTensor ut2 = UniTensor(bonds);
  EXPECT_THROW(ut1.Div_(ut2), std::logic_error);
}

/*=====test info=====
describe:test div two UniTensor, both are only 1 element
====================*/
TEST_F(DenseUniTensorTest, Div__UT1_UT1) {
  int seed = 0;
  UniTensor ut1 = UniTensor({Bond(1)});
  UniTensor ut2 = UniTensor({Bond(1)});
  random::uniform_(ut1, 1, 5.0, seed);
  random::uniform_(ut2, 1, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  ut1.Div_(ut2);
  EXPECT_EQ(clone.at({0}) / ut2.at({0}), ut1.at({0}));
}

/*=====test info=====
describe:test Div_ two UniTensor, one is digonal and the onther is not
====================*/
#if FAIL_CASE_OPEN
TEST_F(DenseUniTensorTest, Div__diag_ndiag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut1 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  is_diag = false;
  auto ut2 = UniTensor(bonds, {"1"}, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut1, 1, 5.0, seed);
  random::uniform_(ut2, 1, 5.0, seed = 1);
  auto ut1_nondiag = ut1.to_dense();
  ut1.Div_(ut2);
  auto ans = ut1_nondiag.Div(ut2);
  EXPECT_TRUE(AreEqUniTensor(ut1, ans));
}
#endif

/*=====test info=====
describe:test Div_ two diagonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, Div__diag_diag) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut1 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  auto ut2 = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut1, 1, 5.0, seed);
  random::uniform_(ut2, 1, 5.0, seed = 1);
  auto clone = ut1.clone();
  auto shape = ut1.shape();
  ut1.Div_(ut2);
  for (cytnx_uint64 i = 0; i < shape[0]; i++) {
    EXPECT_EQ(clone.at({i}) / ut2.at({i}), ut1.at({i}));
  }
}

/*=====test info=====
describe:test Div_ self
====================*/
TEST_F(DenseUniTensorTest, Div__self) {
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  auto ut = UniTensor(bonds);
  int seed = 0;
  random::uniform_(ut, 1, 5.0, seed);
  auto clone = ut.clone();
  auto shape = ut.shape();
  ut.Div_(ut);
  for (cytnx_uint64 i = 0; i < shape[0]; i++) {
    for (cytnx_uint64 j = 0; j < shape[1]; j++) {
      EXPECT_EQ(clone.at({i, j}) / clone.at({i, j}), ut.at({i, j}));
    }
  }
}

/*=====test info=====
describe:test diving two UniTensor with different shape but not contain 1-element UniTesnor.
====================*/
TEST_F(DenseUniTensorTest, Div__UT_UT_rank_error) {
  auto ut1 = UniTensor({Bond(1), Bond(2)});
  auto ut2 = UniTensor({Bond(1), Bond(3)});
  EXPECT_THROW(ut1.Div_(ut2), std::logic_error);
}

/*=====test info=====
describe:test div one UniTensor and one scalar
====================*/
TEST_F(DenseUniTensorTest, Div__UT_Scalar) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    auto cnst = dtype <= Type.ComplexFloat ? Scalar(std::complex<double>(9.1, 2.3)) : Scalar(9);
    cnst = cnst.astype(dtype);
    UniTensor ut = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut, 1, 5.0, seed);
      ut = ut.astype(dtype);
    } else {
      ut = ut.astype(dtype);
      random::uniform_(ut, 1, 5.0, seed);
    }
    auto clone = ut.clone();
    auto shape = ut.shape();
    const double tol = 1.0e-5;
    ut.Div_(cnst);
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          auto diff = abs(clone.at({i, j, k}) / cnst - ut.at({i, j, k}));
          EXPECT_TRUE(diff <= tol);
        }
      }
    }
  }
}

/*=====test info=====
describe:test div one diagonal UniTensor and one scalar
====================*/
TEST_F(DenseUniTensorTest, Div__diagUT_Scalar) {
  auto row_rank = 1u;
  std::vector<std::string> labels = {"1", "2"};
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  bool is_diag = true;
  auto ut = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu, is_diag);
  int seed = 0;
  random::uniform_(ut, 1, 5.0, seed);
  auto cnst = Scalar(int(9));
  auto clone = ut.clone();
  auto shape = ut.shape();
  ut.Div_(cnst);
  EXPECT_TRUE(ut.is_diag());
  for (size_t i = 0; i < shape[0]; i++) {
    EXPECT_DOUBLE_EQ(clone.at<double>({i}) / double(cnst), ut.at<double>({i}));
  }
}

/*=====test info=====
describe:test oprator+=, input two UniTensor
====================*/
TEST_F(DenseUniTensorTest, operatorDiv_UT_UT) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  UniTensor ut1 = UniTensor(bonds);
  UniTensor ut2 = UniTensor(bonds);
  random::uniform_(ut1, 1, 5.0, seed);
  random::uniform_(ut2, 1, 5.0, seed = 1);
  auto ans = ut1.clone();
  ans.Div_(ut2);
  ut1 /= ut2;
  EXPECT_TRUE(AreEqUniTensor(ut1, ans));
}

/*=====test info=====
describe:test oprator+=, input one UniTensor and one Scalar
====================*/
TEST_F(DenseUniTensorTest, operatorDiv_UT_Scalar) {
  std::vector<Bond> bonds = {Bond(3), Bond(4), Bond(2)};
  int seed = 0;
  UniTensor ut = UniTensor(bonds);
  random::uniform_(ut, 1, 5.0, seed);
  auto cnst = Scalar(9.2);
  auto ans = ut.clone();
  ans.Div_(cnst);
  ut /= cnst;
  EXPECT_TRUE(AreEqUniTensor(ut, ans));
}

/*=====test info=====
describe:test Div related function with uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, Div_uninit) {
  EXPECT_ANY_THROW(utzero345.Div(ut_uninit));
  EXPECT_ANY_THROW(utzero345.Div_(ut_uninit));
  EXPECT_ANY_THROW(ut_uninit.Div(utzero345));
  EXPECT_ANY_THROW(ut_uninit.Div_(utzero345));
  EXPECT_ANY_THROW(ut_uninit.Div_(ut_uninit));

  auto cnst = Scalar(9.2);
  EXPECT_ANY_THROW(ut_uninit.Div(cnst));
  EXPECT_ANY_THROW(ut_uninit.Div_(cnst));

  EXPECT_ANY_THROW(ut_uninit /= utzero345);
  EXPECT_ANY_THROW(ut_uninit /= cnst);
}

TEST_F(DenseUniTensorTest, Norm) {
  EXPECT_DOUBLE_EQ(double(utar345.Norm().at({0}).real()), sqrt(59.0 * 60.0 * 119.0 / 6.0));
  EXPECT_DOUBLE_EQ(double(utarcomplex345.Norm().at({0}).real()),
                   sqrt(2.0 * 59.0 * 60.0 * 119.0 / 6.0));
}

/*=====test info=====
describe:test Norm with IntType UniTensor.
====================*/
TEST_F(DenseUniTensorTest, Norm_TypeInt32) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  std::vector<std::string> labels = {"1", "2"};
  auto ut = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu);
  int seed = 0;
  random::uniform_(ut, -5.0, 5.0, seed);
  ut = ut.astype(Type.Int32);
  auto norm = ut.Norm();
  double ans = 0;
  for (cytnx_uint64 i = 0; i < ut.shape()[0]; i++) {
    for (cytnx_uint64 j = 0; j < ut.shape()[1]; j++) {
      ans += static_cast<double>(ut.at<cytnx_int32>({i, j}) * ut.at<cytnx_int32>({i, j}));
    }
  }
  ans = std::sqrt(ans);
  EXPECT_DOUBLE_EQ(norm.at<double>({0}), ans);
}

/*=====test info=====
describe:test Norm with diagonal UniTensor.
====================*/
TEST_F(DenseUniTensorTest, Norm_diag) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  std::vector<std::string> labels = {"1", "2"};
  bool is_diag = true;
  auto seed = 0;
  auto ut_diag = UniTensor(bonds, labels, row_rank, Type.ComplexDouble, Device.cpu, is_diag);
  random::uniform_(ut_diag, -5.0, 5.0, seed);
  auto norm = ut_diag.Norm();
  double ans = 0;
  for (cytnx_uint64 i = 0; i < ut_diag.shape()[0]; i++) {
    ans += std::norm(ut_diag.at<complex<double>>({i}));
  }
  ans = std::sqrt(ans);
  EXPECT_DOUBLE_EQ(norm.at<double>({0}), ans);
}

/*=====test info=====
describe:test Norm with uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, Norm_uninit) { EXPECT_ANY_THROW(ut_uninit.Norm()); }

TEST_F(DenseUniTensorTest, Conj) {
  auto tmp = utarcomplex3456.Conj();
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            // EXPECT_TRUE(Scalar(tmp.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
            EXPECT_DOUBLE_EQ(double(tmp.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(tmp.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             -double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
  tmp = utarcomplex3456.clone();
  utarcomplex3456.Conj_();
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            // EXPECT_TRUE(Scalar(utarcomplex3456.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(tmp.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             -double(tmp.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
}

/*=====test info=====
describe:test Conj with real data type input
====================*/
TEST_F(DenseUniTensorTest, Conj_TypeDouble) {
  std::vector<Bond> bonds = {Bond(3), Bond(2)};
  auto ut = UniTensor(bonds);
  auto seed = 0;
  random::uniform_(ut, -5.0, 5.0, seed);
  auto ut_conj = ut.Conj();
  EXPECT_TRUE(AreEqUniTensor(ut, ut_conj));
}

/*=====test info=====
describe:test Conj with int data type input
====================*/
TEST_F(DenseUniTensorTest, Conj_TypeInt) {
  std::vector<Bond> bonds = {Bond(3), Bond(2)};
  auto ut = UniTensor(bonds);
  auto seed = 0;
  random::uniform_(ut, -5.0, 5.0, seed);
  ut = ut.astype(Type.Int32);
  auto ut_conj = ut.Conj();
  EXPECT_TRUE(AreEqUniTensor(ut, ut_conj));
}

/*=====test info=====
describe:test Conj with diagonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, Conj_diag) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  std::vector<std::string> labels = {"1", "2"};
  bool is_diag = true;
  auto seed = 0;
  auto ut_diag = UniTensor(bonds, labels, row_rank, Type.ComplexDouble, Device.cpu, is_diag);
  random::uniform_(ut_diag, -5.0, 5.0, seed);
  auto ut_conj = ut_diag.Conj();
  for (cytnx_uint64 i = 0; i < ut_diag.shape()[0]; ++i) {
    EXPECT_DOUBLE_EQ(real(ut_diag.at<complex<double>>({i})),
                     real(ut_conj.at<complex<double>>({i})));
    EXPECT_DOUBLE_EQ(imag(ut_diag.at<complex<double>>({i})),
                     -imag(ut_conj.at<complex<double>>({i})));
  }
}

/*=====test info=====
describe:test Conj_
====================*/
TEST_F(DenseUniTensorTest, Conj__) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  std::vector<std::string> labels = {"1", "2"};
  auto seed = 0;
  auto ut = UniTensor(bonds, labels, row_rank, Type.ComplexDouble);
  random::uniform_(ut, -5.0, 5.0, seed);
  auto clone = ut.clone();
  ut.Conj_();
  auto ans = clone.Conj();
  EXPECT_TRUE(AreEqUniTensor(ut, ans));
}
/*=====test info=====
describe:test Conj with uniniitilized UniTensor
====================*/
TEST_F(DenseUniTensorTest, Conj_utuninit) {
  EXPECT_ANY_THROW(ut_uninit.Conj());
  EXPECT_ANY_THROW(ut_uninit.Conj_());
}

/*=====test info=====
describe:test Trnaspose
====================*/
TEST_F(DenseUniTensorTest, Transpose) {
  auto row_rank = 2u;
  std::vector<Bond> bonds = {Bond(3), Bond(2), Bond(2)};
  std::vector<std::string> labels = {"a", "b", "c"};
  auto ut = UniTensor(bonds, labels, row_rank);
  auto seed = 0;
  random::uniform_(ut, -5.0, 5.0, seed);
  auto clone = ut.clone();
  auto ut_t = ut.Transpose();
  for (size_t i = 0; i < ut_t.rank(); i++) {
    EXPECT_EQ(ut_t.bonds()[i].type(), BD_REG);
  }
  // a, b; c -> c;a, b
  EXPECT_EQ(ut.labels(), std::vector<std::string>({"a", "b", "c"}));
  EXPECT_EQ(ut_t.labels(), std::vector<std::string>({"c", "a", "b"}));
  EXPECT_EQ(ut.rowrank(), row_rank);
  EXPECT_EQ(ut_t.rowrank(), ut_t.rank() - row_rank);
  auto shape = ut.shape();
  for (cytnx_uint64 i = 0; i < shape[0]; i++) {
    for (cytnx_uint64 j = 0; j < shape[1]; j++) {
      for (cytnx_uint64 k = 0; k < shape[2]; k++) {
        EXPECT_EQ(ut.at({i, j, k}), ut_t.at({k, i, j}));
      }
    }
  }
  EXPECT_TRUE(AreEqUniTensor(ut, clone));
}

/*=====test info=====
describe:test Trnaspose with diagonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, Transpose_diag) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  std::vector<std::string> labels = {"a", "b"};
  bool is_diag = true;
  auto seed = 0;
  auto ut_diag = UniTensor(bonds, labels, row_rank, Type.ComplexDouble, Device.cpu, is_diag);
  random::uniform_(ut_diag, -5.0, 5.0, seed);
  EXPECT_TRUE(ut_diag.is_diag());
  auto clone = ut_diag.clone();
  auto ut_t = ut_diag.Transpose();
  EXPECT_TRUE(ut_t.is_diag());
  for (size_t i = 0; i < ut_t.rank(); i++) {
    EXPECT_EQ(ut_t.bonds()[i].type(), BD_REG);
  }
  // a, b; c -> c;a, b
  EXPECT_EQ(ut_diag.labels(), std::vector<std::string>({"a", "b"}));
  EXPECT_EQ(ut_t.labels(), std::vector<std::string>({"b", "a"}));
  EXPECT_EQ(ut_diag.rowrank(), row_rank);
  EXPECT_EQ(ut_t.rowrank(), ut_t.rank() - row_rank);
  auto shape = ut_diag.shape();
  for (cytnx_uint64 i = 0; i < shape[0]; i++) {
    EXPECT_EQ(ut_diag.at({i}), ut_t.at({i}));
  }
  EXPECT_TRUE(AreEqUniTensor(ut_diag, clone));
}

/*=====test info=====
describe:test Trnaspose_
====================*/
TEST_F(DenseUniTensorTest, Transpose_) {
  auto row_rank = 2u;
  std::vector<Bond> bonds = {Bond(3), Bond(2), Bond(2)};
  std::vector<std::string> labels = {"a", "b", "c"};
  auto ut = UniTensor(bonds, labels, row_rank);
  auto seed = 0;
  random::uniform_(ut, -5.0, 5.0, seed);
  auto clone = ut.clone();
  ut.Transpose_();
  auto ans = clone.Transpose();
  EXPECT_TRUE(AreEqUniTensor(ut, ans));
}

/*=====test info=====
describe:test Trnaspose with uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, Transpose_uninit) {
  EXPECT_ANY_THROW(ut_uninit.Transpose());
  EXPECT_ANY_THROW(ut_uninit.Transpose_());
}

/*=====test info=====
describe:test normalize
====================*/
TEST_F(DenseUniTensorTest, normalize) {
  auto row_rank = 2u;
  std::vector<Bond> bonds = {Bond(3), Bond(2), Bond(2)};
  std::vector<std::string> labels = {"a", "b", "c"};
  auto ut = UniTensor(bonds, labels, row_rank, Type.ComplexDouble);
  auto seed = 0;
  random::uniform_(ut, -5.0, 5.0, seed);
  auto clone = ut.clone();
  auto ut_n = ut.normalize();
  auto norm = ut.Norm();
  auto ans = ut / norm.at({0});
  EXPECT_TRUE(AreEqUniTensor(ut, clone));
  constexpr double tol = 1.0e-12;
  EXPECT_TRUE(AreNearlyEqUniTensor(ut_n, ans, tol));
}

/*=====test info=====
describe:test normalize
====================*/
TEST_F(DenseUniTensorTest, normalize_int_type) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(2), Bond(2)};
  std::vector<std::string> labels = {"a", "b", "c"};
  auto ut = UniTensor(bonds, labels, row_rank, Type.Int32);
  ut.at({0, 1, 0}) = 1;
  auto clone = ut.clone();
  auto ut_n = ut.normalize();
  auto norm = ut.Norm();
  auto ans = ut / norm.at({0});
  ans = ans.astype(Type.Int32);
  EXPECT_TRUE(AreEqUniTensor(ut, clone));
  EXPECT_TRUE(AreEqUniTensor(ut_n, ans));
}

/*=====test info=====
describe:test normalize with diagonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, normalize_diag) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  std::vector<std::string> labels = {"a", "b"};
  bool is_diag = true;
  auto seed = 0;
  auto ut_diag = UniTensor(bonds, labels, row_rank, Type.ComplexDouble, Device.cpu, is_diag);
  random::uniform_(ut_diag, -5.0, 5.0, seed);
  auto clone = ut_diag.clone();
  auto ut_n = ut_diag.normalize();
  auto norm = ut_diag.Norm();
  auto ans = ut_diag / norm.at({0});
  EXPECT_TRUE(AreEqUniTensor(ut_diag, clone));
  constexpr double tol = 1.0e-12;
  EXPECT_TRUE(AreNearlyEqUniTensor(ut_n, ans, tol));
  EXPECT_TRUE(ut_diag.is_diag());
  EXPECT_TRUE(ut_n.is_diag());
}

/*=====test info=====
describe:test normalize_
====================*/
TEST_F(DenseUniTensorTest, normalize_) {
  auto row_rank = 2u;
  std::vector<Bond> bonds = {Bond(3), Bond(2), Bond(2)};
  std::vector<std::string> labels = {"a", "b", "c"};
  auto ut = UniTensor(bonds, labels, row_rank);
  auto seed = 0;
  random::uniform_(ut, -5.0, 5.0, seed);
  auto clone = ut.clone();
  ut.normalize_();
  auto ans = clone.normalize();
  EXPECT_TRUE(AreEqUniTensor(ut, ans));
}

/*=====test info=====
describe:test normalize with uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, normalize_uninit) {
  EXPECT_ANY_THROW(ut_uninit.normalize());
  EXPECT_ANY_THROW(ut_uninit.normalize_());
}

TEST_F(DenseUniTensorTest, Trace) {
  auto tmp = dense4trtensor.Trace(0, 3);
  for (size_t j = 1; j <= 4; j++)
    for (size_t k = 1; k <= 5; k++)
      if (densetr.at({j - 1, k - 1}).exists()) {
        EXPECT_DOUBLE_EQ(double(tmp.at({j - 1, k - 1}).real()),
                         double(densetr.at({j - 1, k - 1}).real()));
        EXPECT_DOUBLE_EQ(double(tmp.at({j - 1, k - 1}).imag()),
                         double(densetr.at({j - 1, k - 1}).imag()));
      }
  EXPECT_NO_THROW(dense4trtensor.Trace(0, 3));
  EXPECT_THROW(dense4trtensor.Trace(), std::logic_error);
  EXPECT_THROW(dense4trtensor.Trace(0, 1), std::logic_error);
  EXPECT_THROW(dense4trtensor.Trace(-1, 2), std::logic_error);
  EXPECT_THROW(dense4trtensor.Trace(-1, 5), std::logic_error);
}

/*=====test info=====
describe:test Trace with diagonal UniTensor
====================*/
#if FAIL_CASE_OPEN
TEST_F(DenseUniTensorTest, Trace_diag) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  std::vector<std::string> labels = {"a", "b"};
  bool is_diag = true;
  auto seed = 0;
  auto ut_diag = UniTensor(bonds, labels, row_rank, Type.ComplexDouble, Device.cpu, is_diag);
  random::uniform_(ut_diag, -5.0, 5.0, seed);
  auto ut_tr = ut_diag.Trace(0, 1);
  auto ut_dense = ut_diag.to_dense();
  auto ans = ut_dense.Trace(0, 1);
  EXPECT_TRUE(AreEqUniTensor(ut_tr.to_dense(), ans));
}
#endif

/*=====test info=====
describe:test Trace by string label
====================*/
TEST_F(DenseUniTensorTest, Trace_str) {
  auto tmp = dense4trtensor.Trace("0", "3");
  auto ans = dense4trtensor.Trace(0, 3);
  EXPECT_TRUE(AreEqUniTensor(tmp, ans));
}

/*=====test info=====
describe:test Trace_
====================*/
TEST_F(DenseUniTensorTest, Trace_) {
  auto ans = dense4trtensor.Trace(0, 3);
  dense4trtensor.Trace_(0, 3);
  EXPECT_TRUE(AreEqUniTensor(dense4trtensor, ans));
}

/*=====test info=====
describe:test Trace_ by string labels
====================*/
TEST_F(DenseUniTensorTest, Trace__str) {
  auto ans = dense4trtensor.Trace("0", "3");
  dense4trtensor.Trace_("0", "3");
  EXPECT_TRUE(AreEqUniTensor(dense4trtensor, ans));
}

/*=====test info=====
describe:test Trace with uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, Trace_uninit) {
  EXPECT_ANY_THROW(ut_uninit.Trace());
  EXPECT_ANY_THROW(ut_uninit.Trace_());
}

TEST_F(DenseUniTensorTest, Dagger) {
  auto tmp = utzero3456.Dagger();
  EXPECT_EQ(tmp.bonds()[0].type(), BD_REG);
  EXPECT_EQ(tmp.bonds()[1].type(), BD_REG);
  EXPECT_EQ(tmp.bonds()[2].type(), BD_REG);
  EXPECT_EQ(tmp.bonds()[3].type(), BD_REG);

  utzero3456.Dagger_();
  EXPECT_EQ(utzero3456.bonds()[0].type(), BD_REG);
  EXPECT_EQ(utzero3456.bonds()[1].type(), BD_REG);
  EXPECT_EQ(utzero3456.bonds()[2].type(), BD_REG);
  EXPECT_EQ(utzero3456.bonds()[3].type(), BD_REG);

  tmp = utarcomplex3456.Dagger();
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            // EXPECT_TRUE(Scalar(tmp.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
            EXPECT_DOUBLE_EQ(double(tmp.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(tmp.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             -double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
  tmp = utarcomplex3456.clone();
  utarcomplex3456.Dagger_();
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            // EXPECT_TRUE(Scalar(utarcomplex3456.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(tmp.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             -double(tmp.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
}
/*=====test info=====
describe:test Dagger with real date type
====================*/
TEST_F(DenseUniTensorTest, Dagger_real) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(2), Bond(4)};
  std::vector<std::string> labels = {"a", "b", "c"};
  auto seed = 0;
  auto ut = UniTensor(bonds, labels, row_rank, Type.Double);
  random::uniform_(ut, -5.0, 5.0, seed);
  auto ans = ut.Transpose();
  auto dag = ut.Dagger();
  EXPECT_TRUE(AreEqUniTensor(dag, ans));
}

/*=====test info=====
describe:test Dagger with int date type
====================*/
TEST_F(DenseUniTensorTest, Dagger_int) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(2), Bond(4)};
  std::vector<std::string> labels = {"a", "b", "c"};
  auto seed = 0;
  auto ut = UniTensor(bonds, labels, row_rank, Type.Double);
  random::uniform_(ut, -5.0, 5.0, seed);
  ut = ut.astype(Type.Int32);
  auto ans = ut.Transpose();
  auto dag = ut.Dagger();
  EXPECT_TRUE(AreEqUniTensor(dag, ans));
}

/*=====test info=====
describe:test Dagger with diagonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, Dagger_diag) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  std::vector<std::string> labels = {"a", "b"};
  bool is_diag = true;
  auto seed = 0;
  auto ut_diag = UniTensor(bonds, labels, row_rank, Type.ComplexDouble, Device.cpu, is_diag);
  random::uniform_(ut_diag, -5.0, 5.0, seed);
  auto ans = ut_diag.Conj();
  ans = ans.Transpose();
  auto dag = ut_diag.Dagger();
  EXPECT_TRUE(AreEqUniTensor(dag, ans));
}

/*=====test info=====
describe:test Dagger with uninitialization
====================*/
TEST_F(DenseUniTensorTest, Dagger_uninit) {
  EXPECT_ANY_THROW(ut_uninit.Dagger());
  EXPECT_ANY_THROW(ut_uninit.Dagger_());
}

/*=====test info=====
describe:test tag
====================*/
TEST_F(DenseUniTensorTest, tag) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(2), Bond(2)};
  std::vector<std::string> labels = {"a", "b", "c"};
  auto ut = UniTensor(bonds, labels, row_rank);
  auto seed = 0;
  random::uniform_(ut, -5.0, 5.0, seed);
  bonds = ut.bonds();
  EXPECT_EQ(bonds[0].type(), BD_REG);
  EXPECT_EQ(bonds[1].type(), BD_REG);
  EXPECT_EQ(bonds[2].type(), BD_REG);
  EXPECT_FALSE(ut.is_braket_form());
  ut.tag();
  bonds = ut.bonds();
  EXPECT_EQ(bonds[0].type(), BD_KET);
  EXPECT_EQ(bonds[1].type(), BD_BRA);
  EXPECT_EQ(bonds[2].type(), BD_BRA);
  EXPECT_TRUE(ut.is_braket_form());
}

/*=====test info=====
describe:test tag with digonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, tag_diag) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  std::vector<std::string> labels = {"a", "b"};
  bool is_diag = true;
  auto seed = 0;
  auto ut_diag = UniTensor(bonds, labels, row_rank, Type.ComplexDouble, Device.cpu, is_diag);
  random::uniform_(ut_diag, -5.0, 5.0, seed);
  bonds = ut_diag.bonds();
  EXPECT_EQ(bonds[0].type(), BD_REG);
  EXPECT_EQ(bonds[1].type(), BD_REG);
  EXPECT_FALSE(ut_diag.is_braket_form());
  ut_diag.tag();
  bonds = ut_diag.bonds();
  EXPECT_EQ(bonds[0].type(), BD_KET);
  EXPECT_EQ(bonds[1].type(), BD_BRA);
  EXPECT_TRUE(ut_diag.is_braket_form());
}

/*=====test info=====
describe:test tag with uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, tag_uninit) { EXPECT_ANY_THROW(ut_uninit.tag()); }

/*=====test info=====
describe:test pow function of a UniTensor
====================*/
TEST_F(DenseUniTensorTest, Pow) {
  std::vector<Bond> bonds = {Bond(3), Bond(2), Bond(2)};
  int seed = 0;
  for (auto dtype : dtype_list) {
    auto cnst = dtype <= Type.ComplexFloat ? Scalar(std::complex<double>(9.1, 2.3)) : Scalar(9);
    cnst = cnst.astype(dtype);
    UniTensor ut = UniTensor(bonds);
    if (dtype >= Type.Float) {  // not floating type
      random::uniform_(ut, -5, 5.0, seed);
      ut = ut.astype(dtype);
    } else {
      ut = ut.astype(dtype);
      random::uniform_(ut, -5, 5.0, seed);
    }
    auto clone = ut.clone();
    double p = 2.1;
    auto ut_pow = ut.Pow(p);
    auto shape = ut.shape();
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          std::vector<cytnx_uint64> loc = {i, j, k};
          switch (dtype) {
            case Type.ComplexDouble: {
              auto ans = std::pow(clone.at<complex<double>>(loc), p);
              EXPECT_DOUBLE_EQ(static_cast<double>(ut_pow.at(loc).real()), real(ans));
              EXPECT_DOUBLE_EQ(static_cast<double>(ut_pow.at(loc).imag()), imag(ans));
            } break;
            case Type.ComplexFloat: {
              auto ans = std::pow(clone.at<complex<float>>(loc), p);
              EXPECT_FLOAT_EQ(static_cast<float>(ut_pow.at(loc).real()), real(ans));
              EXPECT_FLOAT_EQ(static_cast<float>(ut_pow.at(loc).imag()), imag(ans));
            } break;
            case Type.Double: {
              auto ans = std::pow(clone.at<double>(loc), p);
              auto out = static_cast<double>(ut_pow.at(loc).real());
              if (!(std::isnan(ans) && std::isnan(out))) {
                EXPECT_DOUBLE_EQ(out, ans);
              }
            } break;
            case Type.Float: {
              auto ans = std::pow(clone.at<float>(loc), p);
              auto out = static_cast<float>(ut_pow.at(loc).real());
              if (!(std::isnan(ans) && std::isnan(out))) {
                EXPECT_FLOAT_EQ(out, ans);
              }
            } break;
            case Type.Int64: {
              auto ans = std::pow(clone.at<cytnx_int64>(loc), p);
              auto out = static_cast<double>(ut_pow.at(loc).real());
              if (!(std::isnan(ans) && std::isnan(out))) {
                EXPECT_DOUBLE_EQ(out, ans);
              }
            } break;
            case Type.Uint64: {
              auto ans = std::pow(clone.at<cytnx_uint64>(loc), p);
              auto out = static_cast<double>(ut_pow.at(loc).real());
              if (!(std::isnan(ans) && std::isnan(out))) {
                EXPECT_DOUBLE_EQ(out, ans);
              }
            } break;
            case Type.Int32: {
              auto ans = std::pow(clone.at<cytnx_int32>(loc), p);
              auto out = static_cast<double>(ut_pow.at(loc).real());
              if (!(std::isnan(ans) && std::isnan(out))) {
                EXPECT_DOUBLE_EQ(out, ans);
              }
            } break;
            case Type.Uint32: {
              auto ans = std::pow(clone.at<cytnx_uint32>(loc), p);
              auto out = static_cast<double>(ut_pow.at(loc).real());
              if (!(std::isnan(ans) && std::isnan(out))) {
                EXPECT_DOUBLE_EQ(out, ans);
              }
            } break;
            case Type.Int16: {
              auto ans = std::pow(clone.at<cytnx_int16>(loc), p);
              auto out = static_cast<double>(ut_pow.at(loc).real());
              if (!(std::isnan(ans) && std::isnan(out))) {
                EXPECT_DOUBLE_EQ(out, ans);
              }
            } break;
            case Type.Uint16: {
              auto ans = std::pow(clone.at<cytnx_uint16>(loc), p);
              auto out = static_cast<double>(ut_pow.at(loc).real());
              if (!(std::isnan(ans) && std::isnan(out))) {
                EXPECT_DOUBLE_EQ(out, ans);
              }
            } break;
            case Type.Bool: {
              auto out = static_cast<double>(ut_pow.at(loc).real());
              if (clone.at(loc) == true) {
                EXPECT_EQ(out, 1.0);
              } else {
                EXPECT_EQ(out, 0.0);
              }
            } break;
          }
        }
      }
    }
  }
}

/*=====test info=====
describe:test pow function of a diagonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, Pow_diag) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  std::vector<std::string> labels = {"a", "b"};
  bool is_diag = true;
  auto seed = 0;
  auto ut_diag = UniTensor(bonds, labels, row_rank, Type.ComplexDouble, Device.cpu, is_diag);
  random::uniform_(ut_diag, -5.0, 5.0, seed);
  double p = 0.5;
  auto ut_pow = ut_diag.Pow(p);
  EXPECT_TRUE(ut_pow.is_diag());
  for (cytnx_uint64 i = 0; i < ut_diag.shape()[0]; i++) {
    EXPECT_EQ(ut_pow.at<complex<double>>({i}), std::pow(ut_diag.at<complex<double>>({i}), p));
  }
}

/*=====test info=====
describe:test Pow_
====================*/
TEST_F(DenseUniTensorTest, Pow_) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(2), Bond(3)};
  std::vector<std::string> labels = {"a", "b", "c"};
  auto seed = 0;
  auto ut = UniTensor(bonds, labels, row_rank);
  random::uniform_(ut, 0.0, 5.0, seed);
  double p = 0.5;
  auto clone = ut.clone();
  auto ans = ut.Pow(p);
  ut.Pow_(p);
  EXPECT_TRUE(AreEqUniTensor(ut, ans));
}

/*=====test info=====
describe:test Pow with uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, Pow_uninit) {
  EXPECT_ANY_THROW(ut_uninit.Pow(1));
  EXPECT_ANY_THROW(ut_uninit.Pow_(1));
}

/*=====test info=====
describe:test elem_exists
====================*/
TEST_F(DenseUniTensorTest, elem_exists) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(2)};
  std::vector<std::string> labels = {"a", "b"};
  auto seed = 0;
  auto ut = UniTensor(bonds, labels, row_rank);
  random::uniform_(ut, 0.0, 5.0, seed);
  // only for symmetric UniTensor
  EXPECT_THROW(ut.elem_exists({0, 0}), std::logic_error);
}

/*=====test info=====
describe:test elem_exist with uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, elem_exists_uninit) { EXPECT_ANY_THROW(ut_uninit.elem_exists({0})); }

/*=====test info=====
describe:test Save and Load by string
====================*/
TEST_F(DenseUniTensorTest, Save) {
  const std::string fileName = "SaveUniTestUniTensor";
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(2)};
  std::vector<std::string> labels = {"a", "b"};
  auto seed = 0;
  auto ut = UniTensor(bonds, labels, row_rank);
  random::uniform_(ut, 0.0, 5.0, seed);
  ut.Save(fileName);
  UniTensor ut_load = UniTensor::Load(fileName + ".cytnx");
  EXPECT_TRUE(AreEqUniTensor(ut_load, ut));
}

/*=====test info=====
describe:test Save and Load by charPtr
====================*/
TEST_F(DenseUniTensorTest, Save_chr) {
  const std::string fileName = "SaveUniTestUniTensor_chr";
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(2)};
  std::vector<std::string> labels = {"a", "b"};
  auto seed = 0;
  auto ut = UniTensor(bonds, labels, row_rank);
  random::uniform_(ut, 0.0, 5.0, seed);
  ut.Save(fileName.c_str());
  UniTensor ut_load = UniTensor::Load((fileName + ".cytnx").c_str());
  EXPECT_TRUE(AreEqUniTensor(ut_load, ut));
}

/*=====test info=====
describe:test Save Load not exist file
====================*/
TEST_F(DenseUniTensorTest, Save_path_incorrect) {
  const std::string fileName = "./NotExistFolder/SaveUniTestUniTensor";
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(2)};
  std::vector<std::string> labels = {"a", "b"};
  auto seed = 0;
  auto ut = UniTensor(bonds, labels, row_rank);
  random::uniform_(ut, 0.0, 5.0, seed);
  EXPECT_THROW(ut.Save(fileName), std::logic_error);
  EXPECT_THROW(UniTensor::Load(fileName + ".cytnx"), std::logic_error);
}

/*=====test info=====
describe:test Save uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, Save_uninit) {
  const std::string fileName = "SaveUniTestUniTensor_uninit";
  EXPECT_ANY_THROW(ut_uninit.Save(fileName));
}

/*=====test info=====
describe:test truncate by label
====================*/
TEST_F(DenseUniTensorTest, truncate_label) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(5), Bond(4)};
  std::vector<std::string> labels = {"a", "b"};
  auto seed = 0;
  auto ut = UniTensor(bonds, labels, row_rank);
  random::uniform_(ut, 0.0, 5.0, seed);
  int dim = 3;
  auto clone = ut.clone();
  auto ut_trunc = ut.truncate("a", dim);
  EXPECT_TRUE(AreEqUniTensor(ut, clone));
  auto src_shape = ut.shape();
  auto shape = ut_trunc.shape();
  EXPECT_EQ(shape[0], dim);
  EXPECT_EQ(shape[1], src_shape[1]);
  for (cytnx_uint64 i = 0; i < shape[0]; i++) {
    for (cytnx_uint64 j = 0; j < shape[1]; j++) {
      EXPECT_EQ(ut.at({i, j}), ut_trunc.at({i, j}));
    }
  }
}

/*=====test info=====
describe:test truncate with diangonal UniTensor
====================*/
TEST_F(DenseUniTensorTest, truncate_diag) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(3), Bond(3)};
  std::vector<std::string> labels = {"a", "b"};
  bool is_diag = true;
  auto seed = 0;
  auto ut_diag = UniTensor(bonds, labels, row_rank, Type.ComplexDouble, Device.cpu, is_diag);
  random::uniform_(ut_diag, 0.0, 5.0, seed);
  int dim = 2;
  auto clone = ut_diag.clone();
  auto ut_trunc = ut_diag.truncate("a", dim);
  EXPECT_TRUE(AreEqUniTensor(ut_diag, clone));
  auto shape = ut_trunc.shape();
  EXPECT_EQ(shape[0], dim);
  EXPECT_EQ(shape[1], dim);  // expected ?
  for (cytnx_uint64 i = 0; i < shape[0]; i++) {
    EXPECT_EQ(ut_diag.at({i}), ut_trunc.at({i}));
  }
}

/*=====test info=====
describe:test truncate by label not exist
====================*/
TEST_F(DenseUniTensorTest, truncate_label_not_exist) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(5), Bond(4)};
  std::vector<std::string> labels = {"a", "b"};
  auto seed = 0;
  auto ut = UniTensor(bonds, labels, row_rank);
  random::uniform_(ut, 0.0, 5.0, seed);
  int dim = 3;
  EXPECT_THROW(ut.truncate("c", dim), std::logic_error);
}

/*=====test info=====
describe:test truncate by label but larger dim
====================*/
TEST_F(DenseUniTensorTest, truncate_label_lagerdim) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(2), Bond(3)};
  std::vector<std::string> labels = {"a", "b"};
  auto seed = 0;
  auto ut = UniTensor(bonds, labels, row_rank);
  random::uniform_(ut, 0.0, 5.0, seed);
  int dim = 4;
  EXPECT_THROW(ut.truncate("a", dim), std::logic_error);
  EXPECT_THROW(ut.truncate_("a", dim), std::logic_error);
}

/*=====test info=====
describe:test truncate by index
====================*/
TEST_F(DenseUniTensorTest, truncate_index) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(5), Bond(4)};
  std::vector<std::string> labels = {"a", "b"};
  auto seed = 0;
  auto ut = UniTensor(bonds, labels, row_rank);
  random::uniform_(ut, 0.0, 5.0, seed);
  int dim = 3;
  auto clone = ut.clone();
  auto ut_trunc = ut.truncate(1, dim);
  EXPECT_TRUE(AreEqUniTensor(ut, clone));
  auto src_shape = ut.shape();
  auto shape = ut_trunc.shape();
  EXPECT_EQ(shape[1], dim);
  EXPECT_EQ(shape[0], src_shape[0]);
  for (cytnx_uint64 i = 0; i < shape[0]; i++) {
    for (cytnx_uint64 j = 0; j < shape[1]; j++) {
      EXPECT_EQ(ut.at({i, j}), ut_trunc.at({i, j}));
    }
  }
}

/*=====test info=====
describe:test truncate by index not exist out of range
====================*/
TEST_F(DenseUniTensorTest, truncate_index_outrange) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(5), Bond(4)};
  std::vector<std::string> labels = {"a", "b"};
  auto seed = 0;
  auto ut = UniTensor(bonds, labels, row_rank);
  random::uniform_(ut, 0.0, 5.0, seed);
  int dim = 3;
  EXPECT_THROW(ut.truncate(2, dim), std::logic_error);
}

/*=====test info=====
describe:test truncate_ by label
====================*/
TEST_F(DenseUniTensorTest, truncate__label) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(5), Bond(4)};
  std::vector<std::string> labels = {"a", "b"};
  auto seed = 0;
  auto ut = UniTensor(bonds, labels, row_rank);
  random::uniform_(ut, 0.0, 5.0, seed);
  auto clone = ut.clone();
  int dim = 3;
  auto ans = clone.truncate("a", dim);
  ut.truncate_("a", dim);
  EXPECT_TRUE(AreEqUniTensor(ut, ans));
}

/*=====test info=====
describe:test truncate_ by index
====================*/
TEST_F(DenseUniTensorTest, truncate__index) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {Bond(5), Bond(4)};
  std::vector<std::string> labels = {"a", "b"};
  auto seed = 0;
  auto ut = UniTensor(bonds, labels, row_rank);
  random::uniform_(ut, 0.0, 5.0, seed);
  auto clone = ut.clone();
  int dim = 3;
  auto ans = clone.truncate(1, dim);
  ut.truncate_(1, dim);
  EXPECT_TRUE(AreEqUniTensor(ut, ans));
}

/*=====test info=====
describe:test truncate with uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, truncate_uninit) {
  EXPECT_ANY_THROW(ut_uninit.truncate("", 1));
  EXPECT_ANY_THROW(ut_uninit.truncate(0, 1));
  EXPECT_ANY_THROW(ut_uninit.truncate_("", 1));
  EXPECT_ANY_THROW(ut_uninit.truncate_(0, 1));
}

/*=====test info=====
describe:test get_qinidces
====================*/
TEST_F(DenseUniTensorTest, get_qindices) {
  EXPECT_THROW(utzero345.get_qindices(0), std::logic_error);
}

/*=====test info=====
describe:test get_ito
====================*/
TEST_F(DenseUniTensorTest, get_itoi) { EXPECT_THROW(utzero345.get_itoi(), std::logic_error); }

/*=====test info=====
describe:test zeros_1
====================*/
TEST_F(DenseUniTensorTest, zeros_1d) {
  const cytnx_uint64 Nelem = 5;
  auto ut = UniTensor::zeros(Nelem, {"b"});
  EXPECT_EQ(ut.shape(), std::vector<cytnx_uint64>({Nelem}));
  EXPECT_EQ(ut.rank(), 1);
  EXPECT_EQ(ut.labels(), std::vector<std::string>({"b"}));
  for (cytnx_uint64 i = 0; i < Nelem; ++i) {
    EXPECT_EQ(ut.at({i}), 0);
  }
}

/*=====test info=====
describe:test zeros_1 error, labels more than 2
====================*/
TEST_F(DenseUniTensorTest, zeros_1d_err) {
  EXPECT_THROW(UniTensor::zeros(3, {"a", "b"}), std::logic_error);
}

/*=====test info=====
describe:test zeros
====================*/
TEST_F(DenseUniTensorTest, zeros) {
  std::vector<cytnx_uint64> shape = {5u, 3u, 4u};
  auto ut = UniTensor::zeros(shape);
  EXPECT_EQ(ut.shape(), shape);
  EXPECT_EQ(ut.rank(), shape.size());
  for (cytnx_uint64 i = 0; i < shape[0]; ++i) {
    for (cytnx_uint64 j = 0; j < shape[1]; ++j) {
      for (cytnx_uint64 k = 0; k < shape[2]; ++k) {
        EXPECT_EQ(ut.at({i, j, k}), 0);
      }
    }
  }
}

/*=====test info=====
describe:test zeros erro, label number not match
====================*/
TEST_F(DenseUniTensorTest, zeros_err) {
  std::vector<cytnx_uint64> shape = {3u, 5u, 4u};
  std::vector<std::string> labels = {"a", "b"};
  EXPECT_THROW(UniTensor::zeros(shape, labels), std::logic_error);
}

/*=====test info=====
describe:test ones_1
====================*/
TEST_F(DenseUniTensorTest, ones_1d) {
  const cytnx_uint64 Nelem = 5;
  auto ut = UniTensor::ones(Nelem, {"b"});
  EXPECT_EQ(ut.shape(), std::vector<cytnx_uint64>({Nelem}));
  EXPECT_EQ(ut.rank(), 1);
  EXPECT_EQ(ut.labels(), std::vector<std::string>({"b"}));
  for (cytnx_uint64 i = 0; i < Nelem; ++i) {
    EXPECT_EQ(ut.at({i}), 1);
  }
}

/*=====test info=====
describe:test ones_1 error, labels more than 2
====================*/
TEST_F(DenseUniTensorTest, ones_1d_err) {
  EXPECT_THROW(UniTensor::ones(3, {"a", "b"}), std::logic_error);
}

/*=====test info=====
describe:test ones
====================*/
TEST_F(DenseUniTensorTest, ones) {
  std::vector<cytnx_uint64> shape = {5u, 3u, 4u};
  auto ut = UniTensor::ones(shape);
  EXPECT_EQ(ut.shape(), shape);
  EXPECT_EQ(ut.rank(), shape.size());
  for (cytnx_uint64 i = 0; i < shape[0]; ++i) {
    for (cytnx_uint64 j = 0; j < shape[1]; ++j) {
      for (cytnx_uint64 k = 0; k < shape[2]; ++k) {
        EXPECT_EQ(ut.at({i, j, k}), 1);
      }
    }
  }
}

/*=====test info=====
describe:test ones erro, label number not match
====================*/
TEST_F(DenseUniTensorTest, ones_err) {
  std::vector<cytnx_uint64> shape = {3u, 5u, 4u};
  std::vector<std::string> labels = {"a", "b"};
  EXPECT_THROW(UniTensor::ones(shape, labels), std::logic_error);
}

/*=====test info=====
describe:test arange_1
====================*/
TEST_F(DenseUniTensorTest, arange_step1) {
  const cytnx_uint64 Nelem = 5;
  auto ut = UniTensor::arange(Nelem, {"b"});
  EXPECT_EQ(ut.shape(), std::vector<cytnx_uint64>({Nelem}));
  EXPECT_EQ(ut.rank(), 1);
  EXPECT_EQ(ut.labels(), std::vector<std::string>({"b"}));
  for (cytnx_uint64 i = 0; i < Nelem; ++i) {
    EXPECT_EQ(ut.at({i}), static_cast<double>(i));
  }
}

/*=====test info=====
describe:test arange
====================*/
TEST_F(DenseUniTensorTest, arange) {
  const double start = 0.3, end = -0.2, step = -0.11;
  auto ut = UniTensor::arange(start, end, step);
  EXPECT_EQ(ut.rank(), 1);
  double eps = step < 0 ? -1.0e-12 : 1.0e-12;
  int ans_len = static_cast<int>((end - start + eps) / step) + 1;
  EXPECT_EQ(ut.shape()[0], ans_len);
  for (cytnx_uint64 i = 0; i < ans_len; ++i) {
    double ans = start + step * i;
    EXPECT_EQ(ut.at({i}), ans);
  }
}

/*=====test info=====
describe:test arange_step_error, start < end but step < 0
====================*/
TEST_F(DenseUniTensorTest, arange_step_error) {
  const double start = 0.1, end = 0.7, step = -0.11;
  EXPECT_THROW(UniTensor::arange(start, end, step), std::logic_error);
}

/*=====test info=====
describe:test linspace
====================*/
TEST_F(DenseUniTensorTest, linspace) {
  const double start = 0.3, end = -0.2;
  const cytnx_uint64 Nelem = 7;
  auto ut = UniTensor::linspace(start, end, Nelem);
  EXPECT_EQ(ut.rank(), 1);
  EXPECT_EQ(ut.shape()[0], Nelem);
}

/*=====test info=====
describe:test uniform_1d
====================*/
TEST_F(DenseUniTensorTest, uniform_1d) {
  const cytnx_uint64 Nelem = 7;
  double low = -2.0, high = 3.0;
  auto ut = UniTensor::uniform(Nelem, low, high);
  EXPECT_EQ(ut.rank(), 1);
  EXPECT_EQ(ut.shape()[0], Nelem);
  for (cytnx_uint64 i = 0; i < Nelem; ++i) {
    EXPECT_TRUE(ut.at({i}) <= high);
    EXPECT_TRUE(ut.at({i}) >= low);
  }
}

/*=====test info=====
describe:test uniform
====================*/
TEST_F(DenseUniTensorTest, uniform) {
  std::vector<cytnx_uint64> shape = {5u, 3u, 4u};
  double low = -2.0, high = 3.0;
  auto ut = UniTensor::uniform(shape, low, high);
  EXPECT_EQ(ut.shape(), shape);
  EXPECT_EQ(ut.rank(), shape.size());
  for (cytnx_uint64 i = 0; i < shape[0]; ++i) {
    for (cytnx_uint64 j = 0; j < shape[1]; ++j) {
      for (cytnx_uint64 k = 0; k < shape[2]; ++k) {
        EXPECT_TRUE(ut.at({i, j, k}) <= high);
        EXPECT_TRUE(ut.at({i, j, k}) >= low);
      }
    }
  }
}

/*=====test info=====
describe:test uniform_error, high < low
====================*/
TEST_F(DenseUniTensorTest, uniform_error) {
  std::vector<cytnx_uint64> shape = {5u, 3u, 4u};
  double low = 2.0, high = -3.0;
  EXPECT_THROW(UniTensor::uniform(shape, low, high), std::logic_error);
}

/*=====test info=====
describe:test normal_1d
====================*/
TEST_F(DenseUniTensorTest, normal_1d) {
  const cytnx_uint64 Nelem = 100;
  double mean = 1, std = 0.2;
  auto ut = UniTensor::normal(Nelem, mean, std);
  EXPECT_EQ(ut.rank(), 1);
  EXPECT_EQ(ut.shape()[0], Nelem);
  // just check min < mean < max
  auto min = linalg::Min(ut.get_block());
  auto max = linalg::Max(ut.get_block());
  EXPECT_TRUE(min.at({0}) < mean);
  EXPECT_TRUE(max.at({0}) > mean);
}

/*=====test info=====
describe:test uniform
====================*/
TEST_F(DenseUniTensorTest, normal) {
  std::vector<cytnx_uint64> shape = {5u, 3u, 4u};
  double mean = 1, std = 0.2;
  auto ut = UniTensor::normal(shape, mean, std);
  EXPECT_EQ(ut.shape(), shape);
  EXPECT_EQ(ut.rank(), shape.size());
  // just check min < mean < max
  auto min = linalg::Min(ut.get_block());
  auto max = linalg::Max(ut.get_block());
  EXPECT_TRUE(min.at({0}) < mean);
  EXPECT_TRUE(max.at({0}) > mean);
}

TEST_F(DenseUniTensorTest, identity) {
  UniTensor ut = UniTensor::identity(2, {"row", "col"}, false, Type.Double, Device.cpu);
  EXPECT_EQ(ut.shape().size(), 2);
  EXPECT_EQ(ut.shape()[0], 2);
  EXPECT_EQ(ut.shape()[1], 2);
  EXPECT_EQ(ut.is_contiguous(), true);
  EXPECT_EQ(ut.labels()[0], "row");
  EXPECT_EQ(ut.labels()[1], "col");
  EXPECT_EQ(ut.dtype(), Type.Double);
  EXPECT_EQ(ut.device(), Device.cpu);
  EXPECT_EQ(ut.is_diag(), false);
  EXPECT_DOUBLE_EQ(ut.at<double>({0, 0}), 1);
  EXPECT_DOUBLE_EQ(ut.at<double>({1, 1}), 1);
  EXPECT_DOUBLE_EQ(ut.at<double>({0, 1}), 0);
  EXPECT_DOUBLE_EQ(ut.at<double>({1, 0}), 0);

  ut = UniTensor::identity(2, {"row", "col"}, true, Type.Double, Device.cpu);
  EXPECT_EQ(ut.shape().size(), 2);
  EXPECT_EQ(ut.shape()[0], 2);
  EXPECT_EQ(ut.shape()[1], 2);
  vec_print(cout, ut.labels());
  vec_print(cout, ut.shape());
  EXPECT_EQ(ut.is_contiguous(), true);
  EXPECT_EQ(ut.labels()[0], "row");
  EXPECT_EQ(ut.labels()[1], "col");
  EXPECT_EQ(ut.dtype(), Type.Double);
  EXPECT_EQ(ut.device(), Device.cpu);
  EXPECT_EQ(ut.is_diag(), true);
  std::cout << ut << std::endl;
  EXPECT_DOUBLE_EQ(ut.at<double>({0}), 1);
  EXPECT_DOUBLE_EQ(ut.at<double>({1}), 1);

  ut = UniTensor::identity(3, {"row", "col"}, false, Type.Double, Device.cpu);
  EXPECT_EQ(ut.shape().size(), 2);
  EXPECT_EQ(ut.shape()[0], 3);
  EXPECT_EQ(ut.shape()[1], 3);
  EXPECT_EQ(ut.is_contiguous(), true);
  EXPECT_EQ(ut.labels()[0], "row");
  EXPECT_EQ(ut.labels()[1], "col");
  EXPECT_EQ(ut.dtype(), Type.Double);
  EXPECT_EQ(ut.device(), Device.cpu);
  EXPECT_EQ(ut.is_diag(), false);
  EXPECT_DOUBLE_EQ(ut.at<double>({0, 0}), 1);
  EXPECT_DOUBLE_EQ(ut.at<double>({1, 1}), 1);
  EXPECT_DOUBLE_EQ(ut.at<double>({0, 1}), 0);
  EXPECT_DOUBLE_EQ(ut.at<double>({1, 0}), 0);
  EXPECT_DOUBLE_EQ(ut.at<double>({2, 0}), 0);
  EXPECT_DOUBLE_EQ(ut.at<double>({2, 1}), 0);
  EXPECT_DOUBLE_EQ(ut.at<double>({0, 2}), 0);
  EXPECT_DOUBLE_EQ(ut.at<double>({1, 2}), 0);
  EXPECT_DOUBLE_EQ(ut.at<double>({2, 2}), 1);

  ut = UniTensor::identity(3, {"row", "col"}, true, Type.Double, Device.cpu);
  EXPECT_EQ(ut.shape().size(), 2);
  EXPECT_EQ(ut.shape()[0], 3);
  EXPECT_EQ(ut.shape()[1], 3);
  EXPECT_EQ(ut.is_contiguous(), true);
  EXPECT_EQ(ut.labels()[0], "row");
  EXPECT_EQ(ut.labels()[1], "col");
  EXPECT_EQ(ut.dtype(), Type.Double);
  EXPECT_EQ(ut.device(), Device.cpu);
  EXPECT_EQ(ut.is_diag(), true);
  EXPECT_DOUBLE_EQ(ut.at<double>({0}), 1);
  EXPECT_DOUBLE_EQ(ut.at<double>({1}), 1);
  EXPECT_DOUBLE_EQ(ut.at<double>({2}), 1);
}

TEST_F(DenseUniTensorTest, eye) {
  UniTensor ut = UniTensor::eye(2, {"row", "col"}, false, Type.Double, Device.cpu);
  EXPECT_EQ(ut.shape().size(), 2);
  EXPECT_EQ(ut.shape()[0], 2);
  EXPECT_EQ(ut.shape()[1], 2);
  EXPECT_EQ(ut.is_contiguous(), true);
  EXPECT_EQ(ut.labels()[0], "row");
  EXPECT_EQ(ut.labels()[1], "col");
  EXPECT_EQ(ut.dtype(), Type.Double);
  EXPECT_EQ(ut.device(), Device.cpu);
  EXPECT_EQ(ut.is_diag(), false);
  EXPECT_DOUBLE_EQ(ut.at<double>({0, 0}), 1);
  EXPECT_DOUBLE_EQ(ut.at<double>({1, 1}), 1);
  EXPECT_DOUBLE_EQ(ut.at<double>({0, 1}), 0);
  EXPECT_DOUBLE_EQ(ut.at<double>({1, 0}), 0);

  ut = UniTensor::eye(2, {"row", "col"}, true, Type.Double, Device.cpu);
  EXPECT_EQ(ut.shape().size(), 2);
  EXPECT_EQ(ut.shape()[0], 2);
  EXPECT_EQ(ut.shape()[1], 2);
  vec_print(cout, ut.labels());
  vec_print(cout, ut.shape());
  EXPECT_EQ(ut.is_contiguous(), true);
  EXPECT_EQ(ut.labels()[0], "row");
  EXPECT_EQ(ut.labels()[1], "col");
  EXPECT_EQ(ut.dtype(), Type.Double);
  EXPECT_EQ(ut.device(), Device.cpu);
  EXPECT_EQ(ut.is_diag(), true);
  std::cout << ut << std::endl;
  EXPECT_DOUBLE_EQ(ut.at<double>({0}), 1);
  EXPECT_DOUBLE_EQ(ut.at<double>({1}), 1);

  ut = UniTensor::eye(3, {"row", "col"}, false, Type.Double, Device.cpu);
  EXPECT_EQ(ut.shape().size(), 2);
  EXPECT_EQ(ut.shape()[0], 3);
  EXPECT_EQ(ut.shape()[1], 3);
  EXPECT_EQ(ut.is_contiguous(), true);
  EXPECT_EQ(ut.labels()[0], "row");
  EXPECT_EQ(ut.labels()[1], "col");
  EXPECT_EQ(ut.dtype(), Type.Double);
  EXPECT_EQ(ut.device(), Device.cpu);
  EXPECT_EQ(ut.is_diag(), false);
  EXPECT_DOUBLE_EQ(ut.at<double>({0, 0}), 1);
  EXPECT_DOUBLE_EQ(ut.at<double>({1, 1}), 1);
  EXPECT_DOUBLE_EQ(ut.at<double>({0, 1}), 0);
  EXPECT_DOUBLE_EQ(ut.at<double>({1, 0}), 0);
  EXPECT_DOUBLE_EQ(ut.at<double>({2, 0}), 0);
  EXPECT_DOUBLE_EQ(ut.at<double>({2, 1}), 0);
  EXPECT_DOUBLE_EQ(ut.at<double>({0, 2}), 0);
  EXPECT_DOUBLE_EQ(ut.at<double>({1, 2}), 0);
  EXPECT_DOUBLE_EQ(ut.at<double>({2, 2}), 1);

  ut = UniTensor::eye(3, {"row", "col"}, true, Type.Double, Device.cpu);
  EXPECT_EQ(ut.shape().size(), 2);
  EXPECT_EQ(ut.shape()[0], 3);
  EXPECT_EQ(ut.shape()[1], 3);
  EXPECT_EQ(ut.is_contiguous(), true);
  EXPECT_EQ(ut.labels()[0], "row");
  EXPECT_EQ(ut.labels()[1], "col");
  EXPECT_EQ(ut.dtype(), Type.Double);
  EXPECT_EQ(ut.device(), Device.cpu);
  EXPECT_EQ(ut.is_diag(), true);
  EXPECT_DOUBLE_EQ(ut.at<double>({0}), 1);
  EXPECT_DOUBLE_EQ(ut.at<double>({1}), 1);
  EXPECT_DOUBLE_EQ(ut.at<double>({2}), 1);
}
