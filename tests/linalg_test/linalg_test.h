#ifndef _H_linalg_test
#define _H_linalg_test

#include "cytnx.hpp"
#include <gtest/gtest.h>

using namespace cytnx;

class linalg_bk_Test : public ::testing::Test {
 public:
  // ==================== svd_truncate ===================
  Bond svd_I = Bond(BD_OUT,{Qs(1),Qs(-1)},{1,1});
  Bond svd_J = Bond(BD_IN,{Qs(5),Qs(3),Qs(1),Qs(-1),Qs(-3),Qs(-5),Qs(-7)},{6,22,57,68,38,8,1});
  Bond svd_K = Bond(BD_OUT,{Qs(5),Qs(3),Qs(1),Qs(-1),Qs(-3),Qs(-5),Qs(-7)},{6,22,57,68,38,8,1});
  Bond svd_L = Bond(BD_OUT,{Qs(1),Qs(-1)},{1,1});
  UniTensor svd_T = UniTensor({svd_I,svd_J,svd_K,svd_L},{"a","b","c","d"},1,Type.Double,Device.cpu,false);
  Tensor svd_Sans;
  //==================== Lanczos_Gnd_Ut ===================
  Tensor A = Tensor::Load("../test_data_base/linalg/Lanczos_Gnd/lan_block_A.cytn");
  Tensor B = Tensor::Load("../test_data_base/linalg/Lanczos_Gnd/lan_block_B.cytn");
  Tensor C = Tensor::Load("../test_data_base/linalg/Lanczos_Gnd/lan_block_C.cytn");
  Bond lan_I = Bond(BD_IN,{Qs(-1),Qs(0),Qs(1)},{9,9,9});
  Bond lan_J = Bond(BD_OUT,{Qs(-1),Qs(0),Qs(1)},{9,9,9});
  UniTensor H = UniTensor({lan_I, lan_J});

 protected:
  void SetUp() override {
    //================ svd truncate =======================
    svd_T = svd_T.Load("../test_data_base/linalg/Svd_truncate/Svd_truncate1.cytnx");
    svd_T.set_rowrank(2);
    svd_Sans = Tensor::Load("../test_data_base/linalg/Svd_truncate/S_truncate1.cytn"); 
    svd_Sans = algo::Sort(svd_Sans);
    //==================== Lanczos_Gnd_Ut ===================
    H.put_block(A,0);
    H.put_block(B,1);
    H.put_block(C,2);
    H.set_labels({'a','b'});
  }
  void TearDown() override {}
};

#endif
