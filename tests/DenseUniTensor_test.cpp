#include "DenseUniTensor_test.h"
using namespace std;
using namespace cytnx;
TEST_F(DenseUniTensorTest, Trace) {
  // // std::cout<<utarcomplex3456<<std::endl;
  // auto tmp = utarcomplex3456.Trace(0,3);
  // std::cout<<BUtrT4<<std::endl;
  // std::cout<<tmp<<std::endl;
  // for(size_t j=1;j<=11;j++)
  //     for(size_t k=1;k<=3;k++)
  //       if(BUtrT4.at({j-1,k-1}).exists()){
  //         // EXPECT_TRUE(Scalar(tmp.at({j-1,k-1})-BUtrT4.at({j-1,k-1})).abs()<1e-5);
  //         EXPECT_DOUBLE_EQ(double(tmp.at({j-1,k-1}).real()),double(BUtrT4.at({j-1,k-1}).real()));
  //         EXPECT_DOUBLE_EQ(double(tmp.at({j-1,k-1}).imag()),double(BUtrT4.at({j-1,k-1}).imag()));
  //       }
  // // EXPECT_NO_THROW(utzero3456.Trace(0,3));
  // // EXPECT_THROW(utzero3456.Trace(),std::logic_error);
  // // EXPECT_THROW(utzero3456.Trace(0,1),std::logic_error);
  // // EXPECT_THROW(utzero3456.Trace(-1,2),std::logic_error);
  // // EXPECT_THROW(utzero3456.Trace(-1,5),std::logic_error);
}

TEST_F(DenseUniTensorTest, relabels){
  utzero3456 = utzero3456.relabels({"a", "b", "cd", "d"});
  EXPECT_EQ(utzero3456.labels()[0],"a");
  EXPECT_EQ(utzero3456.labels()[1],"b");
  EXPECT_EQ(utzero3456.labels()[2],"cd");
  EXPECT_EQ(utzero3456.labels()[3],"d");
  utzero3456 = utzero3456.relabels({1,-1,2,1000});
  EXPECT_THROW(utzero3456.relabels({"a","a","b","c"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabels({1,1,0,-1}), std::logic_error);
  EXPECT_THROW(utzero3456.relabels({"a"}), std::logic_error);
  EXPECT_THROW(utzero3456.relabels({1,2}), std::logic_error);
  EXPECT_THROW(utzero3456.relabels({"a","b","c","d","e"}), std::logic_error);
}

TEST_F(DenseUniTensorTest, relabel){
  utzero3456 = utzero3456.relabel("0", "a");
  utzero3456 = utzero3456.relabel("1", "b");
  utzero3456 = utzero3456.relabel("2", "d");
  utzero3456 = utzero3456.relabel("3", "de");
  utzero3456 = utzero3456.relabel("b", "ggg");
  EXPECT_EQ(utzero3456.labels()[0],"a");
  EXPECT_EQ(utzero3456.labels()[1],"ggg");
  EXPECT_EQ(utzero3456.labels()[2],"d");
  EXPECT_EQ(utzero3456.labels()[3],"de");
  utzero3456 = utzero3456.relabel(0,"ccc");
  EXPECT_EQ(utzero3456.labels()[0],"ccc");
  utzero3456 = utzero3456.relabel(0,-1);
  EXPECT_EQ(utzero3456.labels()[0],"-1");
  utzero3456 = utzero3456.relabel(1,-199922);
  EXPECT_EQ(utzero3456.labels()[1],"-199922");
  utzero3456 = utzero3456.relabel("-1","0");
  EXPECT_EQ(utzero3456.labels()[0],"0");
  // utzero3456.relabel(0,'a');
  // EXPECT_EQ(utzero3456.labels()[0],"a");
  EXPECT_THROW(utzero3456.relabel(5,"a"),std::logic_error);
  EXPECT_THROW(utzero3456.relabel(-1,"a"),std::logic_error);
  EXPECT_THROW(utzero3456.relabel(0,"a").relabel(1,"a"),std::logic_error);
  // utzero3456.relabel(0,"a").relabel(1,"a");
  // EXPECT_THROW(utzero3456.relabel("a","b"),std::logic_error);
  // EXPECT_THROW(utzero3456.relabel(5,'a'),std::logic_error);
}

TEST_F(DenseUniTensorTest, Norm){
  EXPECT_DOUBLE_EQ(double(utar345.Norm().at({0}).real()),sqrt(59.0*60.0*119.0/6.0));
  EXPECT_DOUBLE_EQ(double(utarcomplex345.Norm().at({0}).real()),sqrt(2.0*59.0*60.0*119.0/6.0));
}

TEST_F(DenseUniTensorTest, Conj){
  auto tmp = utarcomplex3456.Conj();
  for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(utarcomplex3456.at({i-1,j-1,k-1,l-1}).exists()){
          // EXPECT_TRUE(Scalar(tmp.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
          EXPECT_DOUBLE_EQ(double(tmp.at({i-1,j-1,k-1,l-1}).real()),double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).real()));
          EXPECT_DOUBLE_EQ(double(tmp.at({i-1,j-1,k-1,l-1}).imag()),-double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).imag()));
        }
  tmp = utarcomplex3456;
  utarcomplex3456.Conj_();
  for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(utarcomplex3456.at({i-1,j-1,k-1,l-1}).exists()){
          // EXPECT_TRUE(Scalar(utarcomplex3456.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).real()),double(tmp.at({i-1,j-1,k-1,l-1}).real()));
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).imag()),-double(tmp.at({i-1,j-1,k-1,l-1}).imag()));
        }
}

TEST_F(DenseUniTensorTest, Transpose){
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

TEST_F(DenseUniTensorTest, Dagger){
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
  for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(utarcomplex3456.at({i-1,j-1,k-1,l-1}).exists()){
          // EXPECT_TRUE(Scalar(tmp.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
          EXPECT_DOUBLE_EQ(double(tmp.at({i-1,j-1,k-1,l-1}).real()),double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).real()));
          EXPECT_DOUBLE_EQ(double(tmp.at({i-1,j-1,k-1,l-1}).imag()),-double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).imag()));
        }
  tmp = utarcomplex3456;
  utarcomplex3456.Dagger_();
  for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(utarcomplex3456.at({i-1,j-1,k-1,l-1}).exists()){
          // EXPECT_TRUE(Scalar(utarcomplex3456.at({i-1,j-1,k-1,l-1})-BUconjT4.at({i-1,j-1,k-1,l-1})).abs()<1e-5);
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).real()),double(tmp.at({i-1,j-1,k-1,l-1}).real()));
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).imag()),-double(tmp.at({i-1,j-1,k-1,l-1}).imag()));
        }
}

// TEST_F(DenseUniTensorTest, truncate){
//   auto tmp = utarcomplex3456.truncate(0,1);
//   // EXPECT_EQ(tmp.bonds()[0].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,6}, {0,1}}));
//   // EXPECT_EQ(tmp.bonds()[1].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6}, {0,1}}));
//   tmp = utarcomplex3456.truncate(1,0);
//   // EXPECT_EQ(tmp.bonds()[0].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6}, {0,1}}));
//   // EXPECT_EQ(tmp.bonds()[1].qnums(),std::vector<std::vector<cytnx_int64>>({{1,5}, {1,6}, {0,1}}));
//   utarcomplex3456.truncate_(1,3);
//   // EXPECT_EQ(BUT5.bonds()[0].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6}, {0,1}}));
//   // EXPECT_EQ(BUT5.bonds()[1].qnums(),std::vector<std::vector<cytnx_int64>>({{0,2}, {1,5}, {1,6}}));
//   EXPECT_THROW(utarcomplex3456.truncate(-1,1), std::logic_error);
//   EXPECT_THROW(utarcomplex3456.truncate(0,-1), std::logic_error);
//   EXPECT_THROW(utarcomplex3456.truncate(0,4), std::logic_error);
//   EXPECT_THROW(utarcomplex3456.truncate(2,0), std::logic_error);
//   EXPECT_THROW(utarcomplex3456.truncate_(-1,1), std::logic_error);
//   EXPECT_THROW(utarcomplex3456.truncate_(0,-1), std::logic_error);
//   EXPECT_THROW(utarcomplex3456.truncate_(0,4), std::logic_error);
//   EXPECT_THROW(utarcomplex3456.truncate_(2,0), std::logic_error);
// }

TEST_F(DenseUniTensorTest, Init){
    //different types
    EXPECT_NO_THROW(dut.Init({phy,phy.redirect(),aux},{"a", "b", "c"},1,Type.Float,Device.cpu,false,false));
    EXPECT_NO_THROW(dut.Init({phy,phy.redirect(),aux},{"a", "b", "c"},1,Type.Double,Device.cpu,false,false));
    EXPECT_NO_THROW(dut.Init({phy,phy.redirect(),aux},{"a", "b", "c"},1,Type.ComplexFloat,Device.cpu,false,false));
    EXPECT_NO_THROW(dut.Init({phy,phy.redirect(),aux},{"a", "b", "c"},1,Type.ComplexDouble,Device.cpu,false,false));

    //on gpu device
    // EXPECT_NO_THROW(BkUt.Init({phy,phy.redirect(),aux},{"a", "b", "c"},1,Type.Float,Device.cuda,false,false));
    // EXPECT_NO_THROW(BkUt.Init({phy,phy.redirect(),aux},{"a", "b", "c"},1,Type.Double,Device.cuda,false,false));
    // EXPECT_NO_THROW(BkUt.Init({phy,phy.redirect(),aux},{"a", "b", "c"},1,Type.ComplexFloat,Device.cuda,false,false));
    // EXPECT_NO_THROW(BkUt.Init({phy,phy.redirect(),aux},{"a", "b", "c"},1,Type.ComplexDouble,Device.cuda,false,false));

    //valid rowranks
    EXPECT_ANY_THROW(dut.Init({phy,phy.redirect(),aux},{"a", "b", "c"},99,Type.Float,Device.cpu,false,false));
    EXPECT_NO_THROW(dut.Init({phy,phy.redirect(),aux},{"a", "b", "c"},2,Type.Float,Device.cpu,false,false));
    EXPECT_NO_THROW(dut.Init({phy,phy.redirect(),aux},{"a", "b", "c"},1,Type.Float,Device.cpu,false,false));
    EXPECT_NO_THROW(dut.Init({phy,phy.redirect(),aux},{"a", "b", "c"},-1,Type.Float,Device.cpu,false,false));
    EXPECT_ANY_THROW(dut.Init({phy,phy.redirect(),aux},{"a", "b", "c"},-2,Type.Float,Device.cpu,false,false));
    EXPECT_NO_THROW(dut.Init({phy,phy.redirect(),aux},{"a", "b", "c"},0,Type.Float,Device.cpu,false,false));

    // is_diag = true, but rank>2
    EXPECT_ANY_THROW(dut.Init({phy,phy.redirect(),aux},{"a", "b", "c"},1,Type.Float,Device.cpu,true,false));

    // is_diag = true, but rowrank!=1
    EXPECT_ANY_THROW(dut.Init({phy,phy.redirect()},{"a", "b"},2,Type.Float,Device.cpu,true,false));

    // is_diag = true, but no outward bond
    EXPECT_ANY_THROW(dut.Init({phy,phy},{"a", "b"},1,Type.Float,Device.cpu,true,false));
}

TEST_F(DenseUniTensorTest, Init_by_Tensor){
    // not a valid operation
    EXPECT_ANY_THROW(dut.Init_by_Tensor(tzero345, false, -1));
}


TEST_F(DenseUniTensorTest, shape){
    EXPECT_EQ(std::vector<cytnx::cytnx_uint64>({2,2,1}), Spf.shape());
}


TEST_F(DenseUniTensorTest, dtype) {
    EXPECT_EQ(Spf.dtype(), Type.Float);
    EXPECT_EQ(Spd.dtype(), Type.Double);
    EXPECT_EQ(Spcf.dtype(), Type.ComplexFloat);
    EXPECT_EQ(Spcd.dtype(), Type.ComplexDouble);
}

TEST_F(DenseUniTensorTest, dtype_str) {
    EXPECT_EQ(Spf.dtype_str(), "Float (Float32)");
    EXPECT_EQ(Spd.dtype_str(), "Double (Float64)");
    EXPECT_EQ(Spcf.dtype_str(), "Complex Float (Complex Float32)");
    EXPECT_EQ(Spcd.dtype_str(), "Complex Double (Complex Float64)");
}

TEST_F(DenseUniTensorTest, device) {
    EXPECT_EQ(Spf.device(), Device.cpu);
}

TEST_F(DenseUniTensorTest, device_str){
    EXPECT_EQ(Spf.device_str(), "cytnx device: CPU");
}

TEST_F(DenseUniTensorTest, is_blockform) {
    EXPECT_EQ(Spf.is_blockform(), true);
    EXPECT_EQ(utzero345.is_blockform(), false);
}

TEST_F(DenseUniTensorTest, is_contiguous) {
    EXPECT_EQ(Spf.is_contiguous(), true);
    auto Spf_new = Spf.permute({2,1,0},1,false);
    EXPECT_EQ(Spf_new.is_contiguous(), false);
}

TEST_F(DenseUniTensorTest, set_rowrank) {
    // Spf is a rank-3 tensor
    EXPECT_ANY_THROW(Spf.set_rowrank(-2)); //set_rowrank cannot be negative! 
    EXPECT_ANY_THROW(Spf.set_rowrank(-1));
    EXPECT_NO_THROW(Spf.set_rowrank(0)); 
    EXPECT_NO_THROW(Spf.set_rowrank(1));
    EXPECT_NO_THROW(Spf.set_rowrank(2));
    EXPECT_NO_THROW(Spf.set_rowrank(3));
    EXPECT_ANY_THROW(Spf.set_rowrank(4)); //set_rowrank can only from 0-3 for rank-3 tn
}

TEST_F(DenseUniTensorTest, astype) {
    UniTensor Spf2d = Spf.astype(Type.Double);
    UniTensor Spf2cf = Spf.astype(Type.ComplexFloat);
    UniTensor Spf2cd = Spf.astype(Type.ComplexDouble);
    EXPECT_EQ(Spf.dtype(), Type.Float);
    EXPECT_EQ(Spf2d.dtype(), Type.Double);
    EXPECT_EQ(Spf2cf.dtype(), Type.ComplexFloat);
    EXPECT_EQ(Spf2cd.dtype(), Type.ComplexDouble);
}

TEST_F(DenseUniTensorTest, reshape) {
    EXPECT_ANY_THROW(Spf.reshape({4,1},1));
}

TEST_F(DenseUniTensorTest, reshape_) {
    EXPECT_ANY_THROW(Spf.reshape_({4,1},1));
}

// TEST_F(DenseUniTensorTest, contiguous) {

//     auto bks = UT_pB_ans.permute({1,2,0}).contiguous().get_blocks();
    
//     for(int b = 0;b<bks.size();b++){
//         int ptr = 0;
//         EXPECT_EQ(bks[b].is_contiguous(), true);
//         for(cytnx_uint64 i =0; i<bks[b].shape()[0];i++)
//             for(cytnx_uint64 j =0; j <bks[b].shape()[1];j++)
//                 for(cytnx_uint64 k=0; k <bks[b].shape()[2];k++){
//                 EXPECT_EQ(double(bks[b].at({i,j,k}).real()), bks[b].storage().at<double>(ptr++));
//             }
//     }
// }

// TEST_F(DenseUniTensorTest, contiguous_) {

//     auto tmp = UT_pB_ans.permute({1,2,0});
//     tmp.contiguous_();
//     auto bks = tmp.get_blocks();
    
//     for(int b = 0;b<bks.size();b++){
//         int ptr = 0;
//         EXPECT_EQ(bks[b].is_contiguous(), true);
//         for(cytnx_uint64 i =0; i<bks[b].shape()[0];i++)
//             for(cytnx_uint64 j =0; j <bks[b].shape()[1];j++)
//                 for(cytnx_uint64 k=0; k <bks[b].shape()[2];k++){
//                 EXPECT_EQ(double(bks[b].at({i,j,k}).real()), bks[b].storage().at<double>(ptr++));
//             }
//     }
// }


// TEST_F(DenseUniTensorTest, same_data) {
//     UniTensor B = UT_pB_ans.permute({1,0,2});
//     UniTensor C = B.contiguous();
//     EXPECT_EQ(B.same_data(C), false);
//     EXPECT_EQ(UT_pB_ans.same_data(B), true);
// }

// TEST_F(DenseUniTensorTest, get_blocks) {
//     auto bks = UT_pB_ans.get_blocks();
//     EXPECT_EQ(bks[0].equiv(t0), true);
//     EXPECT_EQ(bks[1].equiv(t1a), true);
//     EXPECT_EQ(bks[2].equiv(t1b), true);
//     EXPECT_EQ(bks[3].equiv(t2), true);
//     // EXPECT_ANY_THROW(UT_pB_ans.get_block({0,0,3}));
// }

// TEST_F(DenseUniTensorTest, get_blocks_) {
//     auto bks = UT_pB_ans.get_blocks_();
//     EXPECT_EQ(bks[0].equiv(t0), true);
//     EXPECT_EQ(bks[1].equiv(t1a), true);
//     EXPECT_EQ(bks[2].equiv(t1b), true);
//     EXPECT_EQ(bks[3].equiv(t2), true);
//     EXPECT_EQ(UT_pB_ans.get_block_(0).same_data(bks[0]), true);
//     EXPECT_EQ(UT_pB_ans.get_block_(1).same_data(bks[1]), true);
//     EXPECT_EQ(UT_pB_ans.get_block_(2).same_data(bks[2]), true);
//     EXPECT_EQ(UT_pB_ans.get_block_(3).same_data(bks[3]), true);
//     // EXPECT_ANY_THROW(UT_pB_ans.get_block({0,0,3}));
// }

// TEST_F(DenseUniTensorTest, clone) {
//     UniTensor cloned = UT_pB_ans.clone();
//     for(size_t i=0;i<5;i++)
//         for(size_t j=0;j<9;j++)
//             for(size_t k=1;k<30;k++){
//                 EXPECT_EQ(cloned.at({i,j,k}).exists(), UT_pB_ans.at({i,j,k}).exists());
//                 if(cloned.at({i,j,k}).exists())
//                     EXPECT_EQ(cloned.at({i,j,k}), UT_pB_ans.at({i,j,k}));
//             }
// }

// TEST_F(DenseUniTensorTest, permute1) {
//     // rank-3 tensor
//     std::vector<cytnx_int64> a = {1,2,0};
//     auto permuted = UT_permute_1.permute(a, -1);
//     for(size_t i=0;i<10;i++)
//         for(size_t j=0;j<6;j++)
//             for(size_t k=0;k<10;k++){
//                 EXPECT_EQ(permuted.at({i,j,k}).exists(), UT_permute_ans1.at({i,j,k}).exists());
//                 if(permuted.at({i,j,k}).exists())
//                     EXPECT_EQ(double(permuted.at({i,j,k}).real()), double(UT_permute_ans1.at({i,j,k}).real()));
//             }
// }

// TEST_F(DenseUniTensorTest, permute2) {
//   std::vector<cytnx_int64> a = {1,0};
//   auto permuted = UT_permute_2.permute(a, -1);

//   for(size_t j=0;j<10;j++)
//     for(size_t k=0;k<10;k++){
//         EXPECT_EQ(permuted.at({j,k}).exists(), UT_permute_ans2.at({j,k}).exists());
//         if(permuted.at({j,k}).exists())
//             EXPECT_EQ(double(permuted.at({j,k}).real()), double(UT_permute_ans2.at({j,k}).real()));
//     }
// }

// TEST_F(DenseUniTensorTest, permute_1) {
//     // rank-3 tensor
//     std::vector<cytnx_int64> a = {1,2,0};
//     auto permuted = UT_permute_1.clone();
//     permuted.permute_(a, -1);
//     for(size_t i=0;i<10;i++)
//         for(size_t j=0;j<6;j++)
//             for(size_t k=0;k<10;k++){
//                 EXPECT_EQ(permuted.at({i,j,k}).exists(), UT_permute_ans1.at({i,j,k}).exists());
//                 if(permuted.at({i,j,k}).exists())
//                     EXPECT_EQ(double(permuted.at({i,j,k}).real()), double(UT_permute_ans1.at({i,j,k}).real()));
//             }
// }

// TEST_F(DenseUniTensorTest, permute_2) {
//     std::vector<cytnx_int64> a = {1,0};
//     auto permuted = UT_permute_2.clone();
//     permuted.permute_(a, -1);
//     for(size_t j=0;j<10;j++)
//         for(size_t k=0;k<10;k++){
//             EXPECT_EQ(permuted.at({j,k}).exists(), UT_permute_ans2.at({j,k}).exists());
//             if(permuted.at({j,k}).exists())
//                 EXPECT_EQ(double(permuted.at({j,k}).real()), double(UT_permute_ans2.at({j,k}).real()));
//         }
// }

TEST_F(DenseUniTensorTest, contract1) {
    ut1.set_labels({"a","b","c","d"});
    ut2.set_labels({"a","aa","bb","cc"});
    UniTensor out = ut1.contract(ut2);
    auto outbk = out.get_block_();
    auto ansbk = contres1.get_block_();
    EXPECT_TRUE(outbk.equiv(ansbk));
}

TEST_F(DenseUniTensorTest, contract2) {
    ut1.set_labels({"a","b","c","d"});
    ut2.set_labels({"a","b","bb","cc"});
    UniTensor out = ut1.contract(ut2);
    auto outbk = out.get_block_();
    auto ansbk = contres2.get_block_();
    EXPECT_TRUE(outbk.equiv(ansbk));
}

TEST_F(DenseUniTensorTest, contract3) {
    ut1.set_labels({"a","b","c","d"});
    ut2.set_labels({"a","b","c","cc"});
    UniTensor out = ut1.contract(ut2);
    auto outbk = out.get_block_();
    auto ansbk = contres3.get_block_();
    EXPECT_TRUE(outbk.equiv(ansbk));
}

TEST_F(DenseUniTensorTest, Add){
    using namespace std::complex_literals;
    auto out = utarcomplex3456.Add(9+9i);
    for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(out.at({i-1,j-1,k-1,l-1}).exists()){
          EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).real()), double(out.at({i-1,j-1,k-1,l-1}).real())+9);
          EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).imag()), double(out.at({i-1,j-1,k-1,l-1}).imag())+9);
        }
    auto tmp = utarcomplex3456;
    utarcomplex3456.Add_(9+9i);
    for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(utarcomplex3456.at({i-1,j-1,k-1,l-1}).exists()){
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).real()), double(tmp.at({i-1,j-1,k-1,l-1}).real())+9);
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).imag()), double(tmp.at({i-1,j-1,k-1,l-1}).imag())+9);
        }
    utarcomplex3456 = UniTensor(arange(3*4*5*6));
    for(size_t i=0;i<3*4*5*6;i++) utarcomplex3456.at({i}) = cytnx_complex128(i,i);
    utarcomplex3456 = utarcomplex3456.reshape({3, 4, 5, 6});
    out = utarcomplex3456.Add(utone3456);
    for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(out.at({i-1,j-1,k-1,l-1}).exists()){
          EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).real()), double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).real()+1));
          EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).imag()), double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).imag()));
        }
    tmp = utarcomplex3456;
    utarcomplex3456.Add_(utone3456);
    for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(utarcomplex3456.at({i-1,j-1,k-1,l-1}).exists()){
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).real()), double(tmp.at({i-1,j-1,k-1,l-1}).real()+1));
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).imag()), double(tmp.at({i-1,j-1,k-1,l-1}).imag()));
        }
}

TEST_F(DenseUniTensorTest, Sub){
    using namespace std::complex_literals;
    auto out = utarcomplex3456.Sub(9+9i);
    for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(out.at({i-1,j-1,k-1,l-1}).exists()){
          EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).real()), double(out.at({i-1,j-1,k-1,l-1}).real())-9);
          EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).imag()), double(out.at({i-1,j-1,k-1,l-1}).imag())-9);
        }
    auto tmp = utarcomplex3456;
    utarcomplex3456.Sub_(9+9i);
    for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(utarcomplex3456.at({i-1,j-1,k-1,l-1}).exists()){
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).real()), double(tmp.at({i-1,j-1,k-1,l-1}).real())-9);
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).imag()), double(tmp.at({i-1,j-1,k-1,l-1}).imag())-9);
        }
    utarcomplex3456 = UniTensor(arange(3*4*5*6));
    for(size_t i=0;i<3*4*5*6;i++) utarcomplex3456.at({i}) = cytnx_complex128(i,i);
    utarcomplex3456 = utarcomplex3456.reshape({3, 4, 5, 6});
    out = utarcomplex3456.Sub(utone3456);
    for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(out.at({i-1,j-1,k-1,l-1}).exists()){
          EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).real()), double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).real()-1));
          EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).imag()), double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).imag()));
        }
    tmp = utarcomplex3456;
    utarcomplex3456.Sub_(utone3456);
    for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(utarcomplex3456.at({i-1,j-1,k-1,l-1}).exists()){
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).real()), double(tmp.at({i-1,j-1,k-1,l-1}).real()-1));
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).imag()), double(tmp.at({i-1,j-1,k-1,l-1}).imag()));
        }
}

TEST_F(DenseUniTensorTest, Mul){
    auto out = utarcomplex3456.Mul(9);
    for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(out.at({i-1,j-1,k-1,l-1}).exists()){
          EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).real()), double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).real()*9));
          EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).imag()), double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).imag()*9));
        }
    out = utarcomplex3456;
    utarcomplex3456.Mul_(9);
    for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(utarcomplex3456.at({i-1,j-1,k-1,l-1}).exists()){
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).real()), double(out.at({i-1,j-1,k-1,l-1}).real()*9));
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).imag()), double(out.at({i-1,j-1,k-1,l-1}).imag()*9));
        }
}

TEST_F(DenseUniTensorTest, Div){
    auto out = utarcomplex3456.Div(9);
    for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(out.at({i-1,j-1,k-1,l-1}).exists()){
          EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).real()), double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).real()/9));
          EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).imag()), double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).imag()/9));
        }
    out = utarcomplex3456;
    utarcomplex3456.Div_(9);
    for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(utarcomplex3456.at({i-1,j-1,k-1,l-1}).exists()){
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).real()), double(out.at({i-1,j-1,k-1,l-1}).real()/9));
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).imag()), double(out.at({i-1,j-1,k-1,l-1}).imag()/9));
        }

    utarcomplex3456 = UniTensor(arange(3*4*5*6));
    for(size_t i=0;i<3*4*5*6;i++) utarcomplex3456.at({i}) = cytnx_complex128(i,i);
    utarcomplex3456 = utarcomplex3456.reshape({3, 4, 5, 6});
    out = utarcomplex3456.Div(utone3456);
    for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(out.at({i-1,j-1,k-1,l-1}).exists()){
          EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).real()), double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).real()));
          EXPECT_DOUBLE_EQ(double(out.at({i-1,j-1,k-1,l-1}).imag()), double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).imag()));
        }
    auto tmp = utarcomplex3456;
    utarcomplex3456.Div_(utone3456);
    for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
      for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
        if(utarcomplex3456.at({i-1,j-1,k-1,l-1}).exists()){
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).real()), double(tmp.at({i-1,j-1,k-1,l-1}).real()));
          EXPECT_DOUBLE_EQ(double(utarcomplex3456.at({i-1,j-1,k-1,l-1}).imag()), double(tmp.at({i-1,j-1,k-1,l-1}).imag()));
        }
}