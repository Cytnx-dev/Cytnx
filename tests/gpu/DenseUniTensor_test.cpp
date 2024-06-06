#include "DenseUniTensor_test.h"
using namespace std;
using namespace cytnx;
using namespace std::complex_literals;
TEST_F(DenseUniTensorTest, gpu_Trace) {
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

TEST_F(DenseUniTensorTest, gpu_relabels) {
  utzero3456 = utzero3456.relabels({"a", "b", "cd", "d"});
  EXPECT_EQ(utzero3456.labels()[0], "a");
  EXPECT_EQ(utzero3456.labels()[1], "b");
  EXPECT_EQ(utzero3456.labels()[2], "cd");
  EXPECT_EQ(utzero3456.labels()[3], "d");
  utzero3456 = utzero3456.relabels({"1", "-1", "2", "1000"});
  EXPECT_THROW(utzero3456.relabels({"a", "a", "b", "c"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabels({"1", "1", "0", "-1"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabels({"a"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabels({"1", "2"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabels({"a", "b", "c", "d", "e"}), std::logic_error);
}
TEST_F(DenseUniTensorTest, gpu_relabels_) {
  utzero3456.relabels_({"a", "b", "cd", "d"});
  EXPECT_EQ(utzero3456.labels()[0], "a");
  EXPECT_EQ(utzero3456.labels()[1], "b");
  EXPECT_EQ(utzero3456.labels()[2], "cd");
  EXPECT_EQ(utzero3456.labels()[3], "d");
  utzero3456.relabels_({"1", "-1", "2", "1000"});
  EXPECT_THROW(utzero3456.relabels_({"a", "a", "b", "c"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabels_({"1", "1", "0", "-1"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabels_({"a"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabels_({"1", "2"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabels_({"a", "b", "c", "d", "e"}), std::logic_error);
}

TEST_F(DenseUniTensorTest, gpu_relabel) {
  auto tmp = utzero3456.clone();
  utzero3456 = utzero3456.relabel({"a", "b", "cd", "d"});
  EXPECT_EQ(utzero3456.labels()[0], "a");
  EXPECT_EQ(utzero3456.labels()[1], "b");
  EXPECT_EQ(utzero3456.labels()[2], "cd");
  EXPECT_EQ(utzero3456.labels()[3], "d");
  utzero3456 = utzero3456.relabel({"1", "-1", "2", "1000"});
  EXPECT_THROW(utzero3456.relabel({"a", "a", "b", "c"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabel({"1", "1", "0", "-1"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabel({"a"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabel({"1", "2"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabel({"a", "b", "c", "d", "e"}), std::logic_error);

  utzero3456 = tmp;
  utzero3456 = utzero3456.relabel("0", "a");
  utzero3456 = utzero3456.relabel("1", "b");
  utzero3456 = utzero3456.relabel("2", "d");
  utzero3456 = utzero3456.relabel("3", "de");
  utzero3456 = utzero3456.relabel("b", "ggg");

  EXPECT_EQ(utzero3456.labels()[0], "a");
  EXPECT_EQ(utzero3456.labels()[1], "ggg");
  EXPECT_EQ(utzero3456.labels()[2], "d");
  EXPECT_EQ(utzero3456.labels()[3], "de");
  utzero3456 = utzero3456.relabel(0, "ccc");
  EXPECT_EQ(utzero3456.labels()[0], "ccc");
  utzero3456 = utzero3456.relabel(0, "-1");
  EXPECT_EQ(utzero3456.labels()[0], "-1");
  utzero3456 = utzero3456.relabel(1, "-199922");
  EXPECT_EQ(utzero3456.labels()[1], "-199922");
  utzero3456 = utzero3456.relabel("-1", "0");
  EXPECT_EQ(utzero3456.labels()[0], "0");

  // utzero3456.relabel(0,'a');
  // EXPECT_EQ(utzero3456.labels()[0],"a");
  EXPECT_THROW(utzero3456.relabel(5, "a"), std::logic_error);
  EXPECT_THROW(utzero3456.relabel(-1, "a"), std::logic_error);
  EXPECT_THROW(utzero3456.relabel(0, "a").relabel(1, "a"), std::logic_error);
  // utzero3456.relabel(0,"a").relabel(1,"a");
  // EXPECT_THROW(utzero3456.relabel("a","b"),std::logic_error);
  // EXPECT_THROW(utzero3456.relabel(5,'a'),std::logic_error);
}
TEST_F(DenseUniTensorTest, gpu_relabel_) {
  auto tmp = utzero3456.clone();
  utzero3456.relabel_({"a", "b", "cd", "d"});
  EXPECT_EQ(utzero3456.labels()[0], "a");
  EXPECT_EQ(utzero3456.labels()[1], "b");
  EXPECT_EQ(utzero3456.labels()[2], "cd");
  EXPECT_EQ(utzero3456.labels()[3], "d");
  utzero3456.relabel_({"1", "-1", "2", "1000"});
  EXPECT_THROW(utzero3456.relabel_({"a", "a", "b", "c"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabel_({"1", "1", "0", "-1"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabel_({"a"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabel_({"1", "2"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabel_({"a", "b", "c", "d", "e"}), std::logic_error);

  utzero3456 = tmp;
  utzero3456.relabel_("0", "a");
  utzero3456.relabel_("1", "b");
  utzero3456.relabel_("2", "d");
  utzero3456.relabel_("3", "de");
  utzero3456.relabel_("b", "ggg");
  EXPECT_EQ(utzero3456.labels()[0], "a");
  EXPECT_EQ(utzero3456.labels()[1], "ggg");
  EXPECT_EQ(utzero3456.labels()[2], "d");
  EXPECT_EQ(utzero3456.labels()[3], "de");
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
  // EXPECT_THROW(utzero3456.relabel_(0,"a").relabel_(1,"a"),std::logic_error);
}

TEST_F(DenseUniTensorTest, gpu_Norm) {
  EXPECT_DOUBLE_EQ(double(utar345.Norm().at({0}).real()), sqrt(59.0 * 60.0 * 119.0 / 6.0));
  EXPECT_DOUBLE_EQ(double(utarcomplex345.Norm().at({0}).real()),
                   sqrt(2.0 * 59.0 * 60.0 * 119.0 / 6.0));
}

TEST_F(DenseUniTensorTest, gpu_Conj) {
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

TEST_F(DenseUniTensorTest, gpu_Transpose) {
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

TEST_F(DenseUniTensorTest, gpu_Dagger) {
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

TEST_F(DenseUniTensorTest, gpu_Init_tagged) {
  // different types
  EXPECT_NO_THROW(dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.Float, Device.cuda,
                           false, false));
  EXPECT_NO_THROW(dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.Double, Device.cuda,
                           false, false));
  EXPECT_NO_THROW(dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.ComplexFloat,
                           Device.cuda, false, false));
  EXPECT_NO_THROW(dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.ComplexDouble,
                           Device.cuda, false, false));

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
  EXPECT_ANY_THROW(dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 99, Type.Float,
                            Device.cuda, false, false));
  EXPECT_NO_THROW(dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 2, Type.Float, Device.cuda,
                           false, false));
  EXPECT_NO_THROW(dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.Float, Device.cuda,
                           false, false));
  EXPECT_NO_THROW(dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, -1, Type.Float, Device.cuda,
                           false, false));
  EXPECT_ANY_THROW(dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, -2, Type.Float,
                            Device.cuda, false, false));
  EXPECT_NO_THROW(dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 0, Type.Float, Device.cuda,
                           false, false));

  // is_diag = true, but rank>2
  EXPECT_ANY_THROW(
    dut.Init({phy, phy.redirect(), aux}, {"a", "b", "c"}, 1, Type.Float, Device.cuda, true, false));

  // is_diag = true, but rowrank!=1
  EXPECT_ANY_THROW(
    dut.Init({phy, phy.redirect()}, {"a", "b"}, 2, Type.Float, Device.cuda, true, false));

  // is_diag = true, but no outward bond
  // cout << phy << endl;
  EXPECT_ANY_THROW(dut.Init({phy, phy}, {"a", "b"}, 1, Type.Float, Device.cuda, true, false));
}

TEST_F(DenseUniTensorTest, gpu_Init_by_Tensor) {
  // EXPECT_NO_THROW(dut.Init_by_Tensor(tar345, false, -1));
  // EXPECT_TRUE(utar345.same_data());
}

TEST_F(DenseUniTensorTest, gpu_shape) {
  EXPECT_EQ(std::vector<cytnx::cytnx_uint64>({2, 2, 1}), Spf.shape());
}

TEST_F(DenseUniTensorTest, gpu_dtype) {
  EXPECT_EQ(Spf.dtype(), Type.Float);
  EXPECT_EQ(Spd.dtype(), Type.Double);
  EXPECT_EQ(Spcf.dtype(), Type.ComplexFloat);
  EXPECT_EQ(Spcd.dtype(), Type.ComplexDouble);
}

TEST_F(DenseUniTensorTest, gpu_dtype_str) {
  EXPECT_EQ(Spf.dtype_str(), "Float (Float32)");
  EXPECT_EQ(Spd.dtype_str(), "Double (Float64)");
  EXPECT_EQ(Spcf.dtype_str(), "Complex Float (Complex Float32)");
  EXPECT_EQ(Spcd.dtype_str(), "Complex Double (Complex Float64)");
}

TEST_F(DenseUniTensorTest, gpu_device) { EXPECT_EQ(Spf.device(), Device.cuda); }

TEST_F(DenseUniTensorTest, gpu_device_str) {
  EXPECT_EQ(Spf.device_str().substr(0, 18), "cytnx device: CUDA");
}

TEST_F(DenseUniTensorTest, gpu_is_blockform) {
  EXPECT_EQ(Spf.is_blockform(), false);
  EXPECT_EQ(utzero345.is_blockform(), false);
}

TEST_F(DenseUniTensorTest, gpu_is_contiguous) {
  EXPECT_EQ(Spf.is_contiguous(), true);
  auto Spf_new = Spf.permute({2, 1, 0}, 1);
  EXPECT_EQ(Spf_new.is_contiguous(), false);
}

TEST_F(DenseUniTensorTest, gpu_set_rowrank) {
  // Spf is a rank-3 tensor
  EXPECT_ANY_THROW(Spf.set_rowrank(-2));  // set_rowrank cannot be negative!
  EXPECT_ANY_THROW(Spf.set_rowrank(-1));
  EXPECT_NO_THROW(Spf.set_rowrank(0));
  EXPECT_NO_THROW(Spf.set_rowrank(1));
  EXPECT_NO_THROW(Spf.set_rowrank(2));
  EXPECT_NO_THROW(Spf.set_rowrank(3));
  EXPECT_ANY_THROW(Spf.set_rowrank(4));  // set_rowrank can only from 0-3 for rank-3 tn
}

TEST_F(DenseUniTensorTest, gpu_astype) {
  UniTensor Spf2d = Spf.astype(Type.Double).to(cytnx::Device.cuda);
  UniTensor Spf2cf = Spf.astype(Type.ComplexFloat).to(cytnx::Device.cuda);
  UniTensor Spf2cd = Spf.astype(Type.ComplexDouble).to(cytnx::Device.cuda);
  EXPECT_EQ(Spf.dtype(), Type.Float);
  EXPECT_EQ(Spf2d.dtype(), Type.Double);
  EXPECT_EQ(Spf2cf.dtype(), Type.ComplexFloat);
  EXPECT_EQ(Spf2cd.dtype(), Type.ComplexDouble);
}

TEST_F(DenseUniTensorTest, gpu_reshape) { EXPECT_ANY_THROW(Spf.reshape({6, 1}, 1)); }

TEST_F(DenseUniTensorTest, gpu_reshape_) { EXPECT_ANY_THROW(Spf.reshape_({6, 1}, 1)); }

TEST_F(DenseUniTensorTest, gpu_contiguous) {
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

TEST_F(DenseUniTensorTest, gpu_contiguous_) {
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

TEST_F(DenseUniTensorTest, gpu_same_data) {
  UniTensor B = ut1.permute({1, 0, 3, 2});
  UniTensor C = B.contiguous();
  EXPECT_FALSE(B.same_data(C));
  EXPECT_TRUE(ut1.same_data(B));
}

TEST_F(DenseUniTensorTest, gpu_get_blocks) {
  EXPECT_THROW(utzero345.get_blocks(), std::logic_error);
}

TEST_F(DenseUniTensorTest, gpu_get_blocks_) {
  EXPECT_THROW(utzero345.get_blocks_(), std::logic_error);
}

TEST_F(DenseUniTensorTest, gpu_clone) {
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

TEST_F(DenseUniTensorTest, gpu_permute1) {
  // rank-4 tensor
  std::vector<cytnx_int64> a = {1, 0, 3, 2};
  auto permuted = ut4.permute(a, -1);
  for (size_t i = 0; i < 4; i++)
    for (size_t j = 0; j < 3; j++)
      for (size_t k = 0; k < 6; k++)
        for (size_t l = 0; l < 5; l++) {
          EXPECT_DOUBLE_EQ(double(permuted.at({i, j, k, l}).real()),
                           double(permu1.at({i, j, k, l}).real()));
          EXPECT_DOUBLE_EQ(double(permuted.at({i, j, k, l}).imag()),
                           double(permu1.at({i, j, k, l}).imag()));
        }
}

TEST_F(DenseUniTensorTest, gpu_permute2) {
  std::vector<cytnx_int64> a = {1, 0};
  auto permuted = ut3.permute(a, -1);
  for (size_t i = 0; i < 6; i++)
    for (size_t j = 0; j < 4; j++) {
      EXPECT_DOUBLE_EQ(double(permuted.at({i, j}).real()), double(permu2.at({i, j}).real()));
      EXPECT_DOUBLE_EQ(double(permuted.at({i, j}).imag()), double(permu2.at({i, j}).imag()));
    }
}

TEST_F(DenseUniTensorTest, gpu_permute_1) {
  // rank-4 tensor
  std::vector<cytnx_int64> a = {1, 0, 3, 2};
  auto permuted = ut4;
  permuted.permute_(a, -1);
  for (size_t i = 0; i < 4; i++)
    for (size_t j = 0; j < 3; j++)
      for (size_t k = 0; k < 6; k++)
        for (size_t l = 0; l < 5; l++) {
          // EXPECT_EQ(complex128(permuted.at({i,j,k,l})), complex128(permu1.at({i,j,k,l})));
          EXPECT_DOUBLE_EQ(double(permuted.at({i, j, k, l}).real()),
                           double(permu1.at({i, j, k, l}).real()));
          EXPECT_DOUBLE_EQ(double(permuted.at({i, j, k, l}).imag()),
                           double(permu1.at({i, j, k, l}).imag()));
        }
}

TEST_F(DenseUniTensorTest, gpu_permute_2) {
  std::vector<cytnx_int64> a = {1, 0};
  auto permuted = ut3;
  permuted.permute_(a, -1);
  for (size_t i = 0; i < 6; i++)
    for (size_t j = 0; j < 4; j++) {
      // EXPECT_EQ(complex128(permuted.at({i,j})), complex128(permu2.at({i,j})));
      EXPECT_DOUBLE_EQ(double(permuted.at({i, j}).real()), double(permu2.at({i, j}).real()));
      EXPECT_DOUBLE_EQ(double(permuted.at({i, j}).imag()), double(permu2.at({i, j}).imag()));
    }
}

TEST_F(DenseUniTensorTest, gpu_contract1) {
  ut1.set_labels({"a", "b", "c", "d"});
  ut2.set_labels({"a", "aa", "bb", "cc"});
  UniTensor out = ut1.contract(ut2);
  auto outbk = out.get_block_();
  auto ansbk = contres1.get_block_();
  EXPECT_TRUE(AreNearlyEqTensor(outbk, ansbk, 1e-5));
}

TEST_F(DenseUniTensorTest, gpu_contract2) {
  ut1.set_labels({"a", "b", "c", "d"});
  ut2.set_labels({"a", "b", "bb", "cc"});
  UniTensor out = ut1.contract(ut2);
  auto outbk = out.get_block_();
  auto ansbk = contres2.get_block_();
  EXPECT_TRUE(AreNearlyEqTensor(outbk, ansbk, 1e-5));
}

TEST_F(DenseUniTensorTest, gpu_contract3) {
  ut1.set_labels({"a", "b", "c", "d"});
  ut2.set_labels({"a", "b", "c", "cc"});
  UniTensor out = ut1.contract(ut2);
  auto outbk = out.get_block_();
  auto ansbk = contres3.get_block_();
  EXPECT_TRUE(AreNearlyEqTensor(outbk, ansbk, 1e-5));
}

TEST_F(DenseUniTensorTest, gpu_Add) {
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
  utarcomplex3456 = UniTensor(arange(3 * 4 * 5 * 6)).astype(Type.ComplexDouble).to(Device.cuda);
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

TEST_F(DenseUniTensorTest, gpu_Sub) {
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
  utarcomplex3456 = UniTensor(arange(3 * 4 * 5 * 6)).astype(Type.ComplexDouble).to(Device.cuda);
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

TEST_F(DenseUniTensorTest, gpu_Mul) {
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

TEST_F(DenseUniTensorTest, gpu_Div) {
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

  utarcomplex3456 = UniTensor(arange(3 * 4 * 5 * 6)).astype(Type.ComplexDouble).to(Device.cuda);
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
