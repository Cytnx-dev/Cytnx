#include "DenseUniTensor_test.h"
using namespace std;
using namespace cytnx;
using namespace std::complex_literals;

#include "test_tools.h"

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
  //too long
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
  EXPECT_EQ(utzero345.Nblocks(), 1); //dense unitensor onely 1 block
  EXPECT_EQ(ut_uninit.Nblocks(), 0); //un-init unitensor
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
TEST_F(DenseUniTensorTest, uten_type) {
  EXPECT_EQ(utzero345.uten_type(), UTenType.Dense);
}

/*=====test info=====
describe:test uten_type for uninitialized tesnor.
====================*/
TEST_F(DenseUniTensorTest, uten_type_uninit) {
  EXPECT_EQ(ut_uninit.uten_type(), UTenType.Void);
}

/*=====test info=====
describe:test dtype. Test for uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, dtype_uninit) {
  EXPECT_ANY_THROW(ut_uninit.dtype());
}

/*=====test info=====
describe:test device.
====================*/
TEST_F(DenseUniTensorTest, device) { EXPECT_EQ(Spf.device(), Device.cpu); }

/*=====test info=====
describe:test device. Test for uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, device_uninit) {
  EXPECT_ANY_THROW(ut_uninit.device());
}

/*=====test info=====
describe:test dtype_str. Test for all possible dypte
====================*/
TEST_F(DenseUniTensorTest, dtype_str) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {phy, phy.redirect(), aux};
  std::vector<std::string> labels = {"1", "2", "3"};
  std::vector<std::string> dtype_str_ans = {
    "Complex Double (Complex Float64)", "Complex Float (Complex Float32)",
    "Double (Float64)", "Float (Float32)", "Int64",
    "Uint64", "Int32", "Uint32", "Int16", "Uint16", "Bool"
  };
  for (size_t i = 0; i < dtype_list.size(); i++) {
    auto dtype = dtype_list[i];
    auto ut = UniTensor(bonds, labels, row_rank, dtype);
    EXPECT_EQ(ut.dtype_str(), dtype_str_ans[i]);
  }
}

/*=====test info=====
describe:test dtype_str. Test for uninitialized UniTensor
====================*/
TEST_F(DenseUniTensorTest, dtype_str_uninit) {
  EXPECT_ANY_THROW(ut_uninit.dtype_str());
}

/*=====test info=====
describe:test uten_type_str for dense tesnor.
====================*/
TEST_F(DenseUniTensorTest, uten_type_str) {
  EXPECT_EQ(utzero345.uten_type_str(), "Dense");
}

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
  auto ut = UniTensor(); //uninitialzie
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

  //default value
  auto ut_diag_default = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu);
  EXPECT_FALSE(ut_diag_default.is_diag());

  //diag false
  bool is_diag = false;
  auto ut_diag_false = UniTensor(bonds, labels, row_rank, 
				                 Type.Double, Device.cpu, is_diag);
  EXPECT_FALSE(ut_diag_false.is_diag());

  //diag true
  is_diag = true;

  auto ut_diag_true = UniTensor(bonds, labels, row_rank, 
				                Type.Double, Device.cpu, is_diag);
  EXPECT_TRUE(ut_diag_true.is_diag());

  //uninitialized
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

  //construct 1-in 1-out unitensor
  auto row_rank = 1u;
  std::vector<Bond> bonds = {phy, phy.redirect()};
  std::vector<std::string> labels = {"1", "2"};
  auto ut = UniTensor(bonds, labels, row_rank, Type.Double, Device.cpu);
  EXPECT_TRUE(ut.is_braket_form());

  //uninitialized
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
TEST_F(DenseUniTensorTest, get_index_uninit) {
  EXPECT_EQ(ut_uninit.get_index(""), -1);
}

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
TEST_F(DenseUniTensorTest, shape_uninit) {
  EXPECT_ANY_THROW(ut_uninit.shape());
}

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

  //uninitialized
  EXPECT_ANY_THROW(ut_uninit.to(Device.cpu));
}

/*=====test info=====
describe:test to_
====================*/
TEST_F(DenseUniTensorTest, to_) {
  Spf.to_(Device.cpu);
  EXPECT_EQ(Spf.device(), Device.cpu);

  //uninitialized
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
  UniTensor ut;
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
  ut = utzero3456.relabel(0,"a").relabel("a","a2");
  EXPECT_EQ(ut.labels()[0], "a2");


  // utzero3456.relabel(0,'a');
  // EXPECT_EQ(utzero3456.labels()[0],"a");
  EXPECT_THROW(utzero3456.relabel(5, "a"), std::logic_error);
  EXPECT_THROW(utzero3456.relabel(-1, "a"), std::logic_error);
  EXPECT_THROW(utzero3456.relabel(0, "a").relabel(1, "a"), std::logic_error);
  EXPECT_THROW(utzero3456.relabel("a","b"),std::logic_error);
  // EXPECT_THROW(utzero3456.relabel(5,'a'),std::logic_error);
  EXPECT_THROW(ut_uninit.relabel(0, ""), std::logic_error);
}
TEST_F(DenseUniTensorTest, relabel_) {
  UniTensor ut;
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
  EXPECT_THROW(utzero3456.relabel_(0,"a").relabel_(1,"a"),std::logic_error);
  EXPECT_THROW(ut_uninit.relabel_(0, ""), std::logic_error);
}

/*=====test info=====
describe:test astype, input all possible dtype.
====================*/
TEST_F(DenseUniTensorTest, astype) {
  auto row_rank = 1u;
  std::vector<Bond> bonds = {phy, phy.redirect()};
  std::vector<std::string> labels = {"1", "2"};

  //from complex double
  auto ut_src = UniTensor(bonds, labels, row_rank, Type.ComplexDouble);
  auto ut_dst = ut_src.astype(Type.ComplexDouble);
  EXPECT_EQ(ut_dst.dtype(), Type.ComplexDouble);
  ut_dst = ut_src.astype(Type.ComplexFloat);
  EXPECT_EQ(ut_dst.dtype(), Type.ComplexFloat);
  EXPECT_THROW(ut_src.astype(Type.Double), std::logic_error); //error test
  EXPECT_THROW(ut_src.astype(Type.Float), std::logic_error); //error test
  EXPECT_THROW(ut_src.astype(Type.Int64), std::logic_error); //error test
  EXPECT_THROW(ut_src.astype(Type.Uint64), std::logic_error); //error test
  EXPECT_THROW(ut_src.astype(Type.Int32), std::logic_error); //error test
  EXPECT_THROW(ut_src.astype(Type.Uint32), std::logic_error); //error test
  EXPECT_THROW(ut_src.astype(Type.Int16), std::logic_error); //error test
  EXPECT_THROW(ut_src.astype(Type.Uint16), std::logic_error); //error test
  EXPECT_THROW(ut_src.astype(Type.Bool), std::logic_error); //error test

  //from complex float
  ut_src = UniTensor(bonds, labels, row_rank, Type.ComplexFloat);
  ut_dst = ut_src.astype(Type.ComplexDouble);
  EXPECT_EQ(ut_dst.dtype(), Type.ComplexDouble);
  ut_dst = ut_src.astype(Type.ComplexFloat);
  EXPECT_EQ(ut_dst.dtype(), Type.ComplexFloat);
  EXPECT_THROW(ut_src.astype(Type.Double), std::logic_error); //error test
  EXPECT_THROW(ut_src.astype(Type.Float), std::logic_error); //error test
  EXPECT_THROW(ut_src.astype(Type.Int64), std::logic_error); //error test
  EXPECT_THROW(ut_src.astype(Type.Uint64), std::logic_error); //error test
  EXPECT_THROW(ut_src.astype(Type.Int32), std::logic_error); //error test
  EXPECT_THROW(ut_src.astype(Type.Uint32), std::logic_error); //error test
  EXPECT_THROW(ut_src.astype(Type.Int16), std::logic_error); //error test
  EXPECT_THROW(ut_src.astype(Type.Uint16), std::logic_error); //error test
  EXPECT_THROW(ut_src.astype(Type.Bool), std::logic_error); //error test

  //from double
  ut_src = UniTensor(bonds, labels, row_rank, Type.Double);
  for(auto dtype_dst : dtype_list) {
    ut_dst = ut_src.astype(dtype_dst);
    EXPECT_EQ(ut_dst.dtype(), dtype_dst);
  }

  //from float
  ut_src = UniTensor(bonds, labels, row_rank, Type.Float);
  for(auto dtype_dst : dtype_list) {
    ut_dst = ut_src.astype(dtype_dst);
    EXPECT_EQ(ut_dst.dtype(), dtype_dst);
  }

  //from int64
  ut_src = UniTensor(bonds, labels, row_rank, Type.Int64);
  for(auto dtype_dst : dtype_list) {
    ut_dst = ut_src.astype(dtype_dst);
    EXPECT_EQ(ut_dst.dtype(), dtype_dst);
  }

  //from uint64
  ut_src = UniTensor(bonds, labels, row_rank, Type.Uint64);
  for(auto dtype_dst : dtype_list) {
    ut_dst = ut_src.astype(dtype_dst);
    EXPECT_EQ(ut_dst.dtype(), dtype_dst);
  }

  //from int32
  ut_src = UniTensor(bonds, labels, row_rank, Type.Int32);
  for(auto dtype_dst : dtype_list) {
    ut_dst = ut_src.astype(dtype_dst);
    EXPECT_EQ(ut_dst.dtype(), dtype_dst);
  }

  //from uint32
  ut_src = UniTensor(bonds, labels, row_rank, Type.Uint32);
  for(auto dtype_dst : dtype_list) {
    ut_dst = ut_src.astype(dtype_dst);
    EXPECT_EQ(ut_dst.dtype(), dtype_dst);
  }

  //from int16
  ut_src = UniTensor(bonds, labels, row_rank, Type.Int16);
  for(auto dtype_dst : dtype_list) {
    ut_dst = ut_src.astype(dtype_dst);
    EXPECT_EQ(ut_dst.dtype(), dtype_dst);
  }

  //from uint16
  ut_src = UniTensor(bonds, labels, row_rank, Type.Uint16);
  for(auto dtype_dst : dtype_list) {
    ut_dst = ut_src.astype(dtype_dst);
    EXPECT_EQ(ut_dst.dtype(), dtype_dst);
  }

  //from Bool
  ut_src = UniTensor(bonds, labels, row_rank, Type.Bool);
  for(auto dtype_dst : dtype_list) {
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
  for (auto dtype:dtype_list) {
    UniTensor ut = UniTensor(bonds);
	if (dtype >= Type.Float) { //not floating type
      random::Make_uniform(ut, -100.0, 100.0, seed);
	  ut = ut.astype(dtype);
	} else {
	  ut = ut.astype(dtype);
      random::Make_uniform(ut, -100.0, 100.0, seed);
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
  random::Make_uniform(ut, -100.0, 100.0, seed);
  std::vector<cytnx_int64> map = {0, 1, 2};
  auto permuted = ut.permute(map, 1);
  EXPECT_EQ(permuted.rowrank(), 1);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 5; k++) {
          EXPECT_DOUBLE_EQ(double(ut.at({i, j, k}).real()),
                           double(permuted.at({i, j, k}).real()));
          EXPECT_DOUBLE_EQ(double(ut.at({i, j, k}).imag()),
                           double(permuted.at({i, j, k}).imag()));
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
  random::Make_uniform(ut, -100.0, 100.0, seed);
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
    EXPECT_DOUBLE_EQ(double(ut_complex_diag.at({i}).real()),
                     double(permuted.at({i}).real()));
    EXPECT_DOUBLE_EQ(double(ut_complex_diag.at({i}).imag()),
                     double(permuted.at({i}).imag()));
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
  random::Make_uniform(ut, -100.0, 100.0, seed);
  std::vector<std::string> map = {"2", "0", "1"};
  auto permuted = ut.permute(map, 1);
  EXPECT_EQ(permuted.rowrank(), 1);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 5; k++) {
          EXPECT_DOUBLE_EQ(double(ut.at({i, j, k}).real()),
                           double(permuted.at({k, i, j}).real()));
          EXPECT_DOUBLE_EQ(double(ut.at({i, j, k}).imag()),
                           double(permuted.at({k, i, j}).imag()));
  	  }
    }
  }
}

TEST_F(DenseUniTensorTest, permute_err) {
  EXPECT_THROW(utzero345.permute({1, 2}, 0), std::logic_error);
  EXPECT_THROW(utzero345.permute({2, 3, 1}, 0), std::logic_error);
  EXPECT_THROW(utzero345.permute({}, 0), std::logic_error);

  //for diag UniTensor, rowrank need to be 1.
  EXPECT_THROW(ut_complex_diag.permute(std::vector<cytnx_int64>({1, 0}), 0), 
               std::logic_error);
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
  for (auto dtype:dtype_list) {
    UniTensor ut = UniTensor(bonds);
	if (dtype >= Type.Float) { //not floating type
      random::Make_uniform(ut, -100.0, 100.0, seed);
	  ut = ut.astype(dtype);
	} else {
	  ut = ut.astype(dtype);
      random::Make_uniform(ut, -100.0, 100.0, seed);
	}
    std::vector<cytnx_int64> map = {1, 0, 2};
    UniTensor src = ut.clone();
    ut.permute_(map, 1);
    EXPECT_EQ(ut.rowrank(), 1);
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 4; j++) {
        for (size_t k = 0; k < 5; k++) {
		  if (dtype == Type.ComplexDouble || dtype == Type.ComplexFloat) {
            EXPECT_DOUBLE_EQ(double(src.at({i, j, k}).real()),
                             double(ut.at({j, i, k}).real()));
            EXPECT_DOUBLE_EQ(double(src.at({i, j, k}).imag()),
                             double(ut.at({j, i, k}).imag()));
		  } else {
            EXPECT_DOUBLE_EQ(double(src.at({i, j, k}).real()),
                             double(ut.at({j, i, k}).real()));
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
  random::Make_uniform(ut, -100.0, 100.0, seed);
  std::vector<cytnx_int64> map = {0, 1, 2};
  auto src = ut.clone();
  ut.permute_(map, 1);
  EXPECT_EQ(ut.rowrank(), 1);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 5; k++) {
          EXPECT_DOUBLE_EQ(double(src.at({i, j, k}).real()),
                           double(ut.at({i, j, k}).real()));
          EXPECT_DOUBLE_EQ(double(src.at({i, j, k}).imag()),
                           double(ut.at({i, j, k}).imag()));
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
  random::Make_uniform(ut, -100.0, 100.0, seed);
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
    EXPECT_DOUBLE_EQ(double(ut_complex_diag.at({i}).real()),
                     double(src.at({i}).real()));
    EXPECT_DOUBLE_EQ(double(ut_complex_diag.at({i}).imag()),
                     double(src.at({i}).imag()));
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
  random::Make_uniform(ut, -100.0, 100.0, seed);
  std::vector<std::string> map = {"2", "0", "1"};
  auto src = ut.clone();
  ut.permute_(map, 1);
  EXPECT_EQ(ut.rowrank(), 1);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 5; k++) {
          EXPECT_DOUBLE_EQ(double(src.at({i, j, k}).real()),
                           double(ut.at({k, i, j}).real()));
          EXPECT_DOUBLE_EQ(double(src.at({i, j, k}).imag()),
                           double(ut.at({k, i, j}).imag()));
  	  }
    }
  }
}

TEST_F(DenseUniTensorTest, permute__err) {
  auto ut = utzero345.clone();
  EXPECT_THROW(ut.permute({1, 2}, 0), std::logic_error);
  EXPECT_THROW(ut.permute({2, 3, 1}, 0), std::logic_error);
  EXPECT_THROW(ut.permute({}, 0), std::logic_error);

  //for diag UniTensor, rowrank need to be 1.
  EXPECT_THROW(ut_complex_diag.permute_(std::vector<cytnx_int64>({1, 0}), 0), 
               std::logic_error);
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
  random::Make_uniform(utzero345, -100.0, 100.0, seed);
  auto ut_grp = utzero345.group_basis();
  EXPECT_TRUE(AreEqUniTensor(utzero345, ut_grp));
}

/*=====test info=====
describe:test group basis_
====================*/
TEST_F(DenseUniTensorTest, group_basis_) {
  int seed = 0;
  random::Make_uniform(utzero345, -100.0, 100.0, seed);
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
  for(auto dtype:dtype_list) {
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
		}
	    break;
	  case Type.ComplexFloat: {
		  ut = ut.astype(dtype);
  	      auto elem = complex<float>(1, -1);
		  ut.at(loc) = elem;
		  EXPECT_EQ(ut.at(loc), elem);
		  EXPECT_EQ(ut.at<complex<float>>(loc), elem);
		  EXPECT_EQ(cut.at(loc), complex<float>());
		  EXPECT_EQ(cut.at<complex<float>>(loc), complex<float>());
		}
	    break;
	  case Type.Double: {
		  ut = ut.astype(dtype);
  	      auto elem = double(1);
		  ut.at(loc) = elem;
		  EXPECT_EQ(ut.at(loc), elem);
		  EXPECT_EQ(ut.at<double>(loc), elem);
		  EXPECT_EQ(cut.at(loc), double());
		  EXPECT_EQ(cut.at<double>(loc), double());
		}
	    break;
	  case Type.Float: {
		  ut = ut.astype(dtype);
  	      auto elem = float(1);
		  ut.at(loc) = elem;
		  EXPECT_EQ(ut.at(loc), elem);
		  EXPECT_EQ(ut.at<float>(loc), elem);
		  EXPECT_EQ(cut.at(loc), float());
		  EXPECT_EQ(cut.at<float>(loc), float());
		}
	    break;
	  case Type.Int64: {
		  ut = ut.astype(dtype);
  	      auto elem = cytnx_int64(1);
		  ut.at(loc) = elem;
		  EXPECT_EQ(ut.at(loc), elem);
		  EXPECT_EQ(ut.at<cytnx_int64>(loc), elem);
		  EXPECT_EQ(cut.at(loc), cytnx_int64());
		  EXPECT_EQ(cut.at<cytnx_int64>(loc), cytnx_int64());
		}
	    break;
	  case Type.Uint64: {
		  ut = ut.astype(dtype);
  	      auto elem = cytnx_uint64(1);
		  ut.at(loc) = elem;
		  EXPECT_EQ(ut.at(loc), elem);
		  EXPECT_EQ(ut.at<cytnx_uint64>(loc), elem);
		  EXPECT_EQ(cut.at(loc), cytnx_uint64());
		  EXPECT_EQ(cut.at<cytnx_uint64>(loc), cytnx_uint64());
		}
	    break;
	  case Type.Int32: {
		  ut = ut.astype(dtype);
  	      auto elem = cytnx_int32(1);
		  ut.at(loc) = elem;
		  EXPECT_EQ(ut.at(loc), elem);
		  EXPECT_EQ(ut.at<cytnx_int32>(loc), elem);
		  EXPECT_EQ(cut.at(loc), cytnx_int32());
		  EXPECT_EQ(cut.at<cytnx_int32>(loc), cytnx_int32());
		}
	    break;
	  case Type.Uint32: {
		  ut = ut.astype(dtype);
  	      auto elem = cytnx_uint32(1);
		  ut.at(loc) = elem;
		  EXPECT_EQ(ut.at(loc), elem);
		  EXPECT_EQ(ut.at<cytnx_uint32>(loc), elem);
		  EXPECT_EQ(cut.at(loc), cytnx_uint32());
		  EXPECT_EQ(cut.at<cytnx_uint32>(loc), cytnx_uint32());
		}
	    break;
	  case Type.Int16: {
		  ut = ut.astype(dtype);
  	      auto elem = cytnx_int16(1);
		  ut.at(loc) = elem;
		  EXPECT_EQ(ut.at(loc), elem);
		  EXPECT_EQ(ut.at<cytnx_int16>(loc), elem);
		  EXPECT_EQ(cut.at(loc), cytnx_int16());
		  EXPECT_EQ(cut.at<cytnx_int16>(loc), cytnx_int16());
		}
	    break;
	  case Type.Uint16: {
		  ut = ut.astype(dtype);
  	      auto elem = cytnx_uint16(1);
		  ut.at(loc) = elem;
		  EXPECT_EQ(ut.at(loc), elem);
		  EXPECT_EQ(ut.at<cytnx_uint16>(loc), elem);
		  EXPECT_EQ(cut.at(loc), cytnx_uint16());
		  EXPECT_EQ(cut.at<cytnx_uint16>(loc), cytnx_uint16());
		}
	    break;
	  case Type.Bool: {
		  ut = ut.astype(dtype);
  	      auto elem = true;
		  ut.at(loc) = elem;
		  EXPECT_EQ(ut.at(loc), elem);
		  EXPECT_EQ(cut.at(loc), bool());
		}
	    break;
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
  random::Make_uniform(tens, -100.0, 100.0, seed);
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
  random::Make_uniform(tens, -100.0, 100.0, seed);
  auto ut_src = UniTensor(tens);
  auto bk = ut_src.get_block();
  auto bk0 = ut_src.get_block(0);
  EXPECT_TRUE(AreEqTensor(bk, bk0));
  EXPECT_THROW(ut_src.get_block(1), std::logic_error);
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
  random::Make_uniform(tens, -100.0, 100.0, seed);
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
  random::Make_uniform(tens, -100.0, 100.0, seed);
  auto ut_src = UniTensor(tens);
  const UniTensor cut_src = UniTensor(tens).clone();
  auto bk = ut_src.get_block_();
  auto bk0 = ut_src.get_block_(0);
  auto bkc = cut_src.get_block_(0);
  EXPECT_TRUE(AreEqTensor(bk, bk0));
  EXPECT_TRUE(AreEqTensor(bk, bkc));
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
  EXPECT_THROW(ut_uninit.get_block({1, 2}), std::logic_error);
}

TEST_F(DenseUniTensorTest, get_blocks) { EXPECT_THROW(utzero345.get_blocks(), std::logic_error); }

TEST_F(DenseUniTensorTest, get_blocks_) { 
  const UniTensor cut = utzero345.clone();
  EXPECT_THROW(utzero345.get_blocks_(), std::logic_error); 
  EXPECT_THROW(cut.get_blocks_(), std::logic_error); 
}


TEST_F(DenseUniTensorTest, reshape) { EXPECT_ANY_THROW(Spf.reshape({6, 1}, 1)); }

TEST_F(DenseUniTensorTest, reshape_) { EXPECT_ANY_THROW(Spf.reshape_({6, 1}, 1)); }


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

TEST_F(DenseUniTensorTest, Add) {
  auto cnst = Scalar(std::complex<double>(9, 9));
  auto out = utarcomplex3456.Add(cnst);
  // cout << Scalar(std::complex<double>(9,9)) << endl;
  // cout << out;
  // cout << utarcomplex3456;
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (out.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real()) + 9);
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag()) + 9);
          }
  auto tmp = utarcomplex3456.clone();
  utarcomplex3456.Add_(cnst);
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(tmp.at({i - 1, j - 1, k - 1, l - 1}).real()) + 9);
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(tmp.at({i - 1, j - 1, k - 1, l - 1}).imag()) + 9);
          }
  utarcomplex3456 = UniTensor(arange(3 * 4 * 5 * 6)).astype(Type.ComplexDouble);
  for (size_t i = 0; i < 3 * 4 * 5 * 6; i++) utarcomplex3456.at({i}) = cytnx_complex128(i, i);
  utarcomplex3456 = utarcomplex3456.reshape({3, 4, 5, 6});
  out = utarcomplex3456.Add(utone3456);
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (out.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real() + 1));
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
  tmp = utarcomplex3456.clone();
  utarcomplex3456.Add_(utone3456);
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(tmp.at({i - 1, j - 1, k - 1, l - 1}).real() + 1));
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(tmp.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
}
TEST_F(DenseUniTensorTest, Mul) {
  auto out = utarcomplex3456.Mul(9);
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (out.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real() * 9));
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag() * 9));
          }
  out = utarcomplex3456.clone();
  utarcomplex3456.Mul_(9);
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(out.at({i - 1, j - 1, k - 1, l - 1}).real() * 9));
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(out.at({i - 1, j - 1, k - 1, l - 1}).imag() * 9));
          }
}

TEST_F(DenseUniTensorTest, Sub) {
  auto cnst = Scalar(std::complex<double>(9, 9));
  auto out = utarcomplex3456.Sub(cnst);
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (out.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real()) - 9);
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag()) - 9);
          }
  auto tmp = utarcomplex3456.clone();
  utarcomplex3456.Sub_(cnst);
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(tmp.at({i - 1, j - 1, k - 1, l - 1}).real()) - 9);
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(tmp.at({i - 1, j - 1, k - 1, l - 1}).imag()) - 9);
          }
  utarcomplex3456 = UniTensor(arange(3 * 4 * 5 * 6)).astype(Type.ComplexDouble);
  for (size_t i = 0; i < 3 * 4 * 5 * 6; i++) utarcomplex3456.at({i}) = cytnx_complex128(i, i);
  utarcomplex3456 = utarcomplex3456.reshape({3, 4, 5, 6});
  out = utarcomplex3456.Sub(utone3456);
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (out.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real() - 1));
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
  tmp = utarcomplex3456.clone();
  utarcomplex3456.Sub_(utone3456);
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(tmp.at({i - 1, j - 1, k - 1, l - 1}).real() - 1));
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(tmp.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
}


TEST_F(DenseUniTensorTest, Div) {
  auto out = utarcomplex3456.Div(9);
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (out.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real() / 9));
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag() / 9));
          }
  out = utarcomplex3456.clone();
  utarcomplex3456.Div_(9);
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(out.at({i - 1, j - 1, k - 1, l - 1}).real() / 9));
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(out.at({i - 1, j - 1, k - 1, l - 1}).imag() / 9));
          }

  utarcomplex3456 = UniTensor(arange(3 * 4 * 5 * 6)).astype(Type.ComplexDouble);
  for (size_t i = 0; i < 3 * 4 * 5 * 6; i++) utarcomplex3456.at({i}) = cytnx_complex128(i, i);
  utarcomplex3456 = utarcomplex3456.reshape({3, 4, 5, 6});
  out = utarcomplex3456.Div(utone3456);
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (out.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(out.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
  auto tmp = utarcomplex3456.clone();
  utarcomplex3456.Div_(utone3456);
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          if (utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).exists()) {
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).real()),
                             double(tmp.at({i - 1, j - 1, k - 1, l - 1}).real()));
            EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i - 1, j - 1, k - 1, l - 1}).imag()),
                             double(tmp.at({i - 1, j - 1, k - 1, l - 1}).imag()));
          }
}

TEST_F(DenseUniTensorTest, Norm) {
  EXPECT_DOUBLE_EQ(double(utar345.Norm().at({0}).real()), sqrt(59.0 * 60.0 * 119.0 / 6.0));
  EXPECT_DOUBLE_EQ(double(utarcomplex345.Norm().at({0}).real()),
                   sqrt(2.0 * 59.0 * 60.0 * 119.0 / 6.0));
}
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

TEST_F(DenseUniTensorTest, Transpose) {
  auto tmp = utzero3456.Transpose();
  EXPECT_EQ(tmp.bonds()[0].type(), BD_REG);
  EXPECT_EQ(tmp.bonds()[1].type(), BD_REG);
  EXPECT_EQ(tmp.bonds()[2].type(), BD_REG);
  EXPECT_EQ(tmp.bonds()[3].type(), BD_REG);

  utzero3456.Transpose_();
  EXPECT_EQ(utzero3456.bonds()[0].type(), BD_REG);
  EXPECT_EQ(utzero3456.bonds()[1].type(), BD_REG);
  EXPECT_EQ(utzero3456.bonds()[2].type(), BD_REG);
  EXPECT_EQ(utzero3456.bonds()[3].type(), BD_REG);
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
  // EXPECT_NO_THROW(utzero3456.Trace(0,3));
  // EXPECT_THROW(utzero3456.Trace(),std::logic_error);
  // EXPECT_THROW(utzero3456.Trace(0,1),std::logic_error);
  // EXPECT_THROW(utzero3456.Trace(-1,2),std::logic_error);
  // EXPECT_THROW(utzero3456.Trace(-1,5),std::logic_error);
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

// TEST_F(DenseUniTensorTest, truncate){
//   auto tmp = utarcomplex3456.truncate(0,1);
//   // EXPECT_EQ(tmp.bonds()[0].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,6},
//   {0,1}}));
//   // EXPECT_EQ(tmp.bonds()[1].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6},
//   {0,1}})); tmp = utarcomplex3456.truncate(1,0);
//   // EXPECT_EQ(tmp.bonds()[0].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6},
//   {0,1}}));
//   // EXPECT_EQ(tmp.bonds()[1].qnums(),std::vector<std::vector<cytnx_int64>>({{1,5}, {1,6},
//   {0,1}})); utarcomplex3456.truncate_(1,3);
//   // EXPECT_EQ(BUT5.bonds()[0].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5},
//   {1,6}, {0,1}}));
//   // EXPECT_EQ(BUT5.bonds()[1].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5},
//   {1,6}})); EXPECT_THROW(utarcomplex3456.truncate(-1,1), std::logic_error);
//   EXPECT_THROW(utarcomplex3456.truncate(0,-1), std::logic_error);
//   EXPECT_THROW(utarcomplex3456.truncate(0,4), std::logic_error);
//   EXPECT_THROW(utarcomplex3456.truncate(2,0), std::logic_error);
//   EXPECT_THROW(utarcomplex3456.truncate_(-1,1), std::logic_error);
//   EXPECT_THROW(utarcomplex3456.truncate_(0,-1), std::logic_error);
//   EXPECT_THROW(utarcomplex3456.truncate_(0,4), std::logic_error);
//   EXPECT_THROW(utarcomplex3456.truncate_(2,0), std::logic_error);
// }


