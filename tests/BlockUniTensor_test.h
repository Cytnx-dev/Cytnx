#ifndef _H_BlockUniTensor_test
#define _H_BlockUniTensor_test

#include "cytnx.hpp"
#include <gtest/gtest.h>

using namespace cytnx;

class BlockUniTensorTest : public ::testing::Test {
 public:
  Bond B1 = Bond(BD_IN, {Qs(0)>>1, Qs(1)>>2});
  Bond B2 = Bond(BD_IN, {Qs(0), Qs(1)}, {3, 4});
  Bond B3 = Bond(BD_OUT, {Qs(0)>>2, Qs(1)>>3});
  Bond B4 = Bond(BD_OUT, {Qs(0), Qs(1)}, {1, 2});
  UniTensor BUT1 = UniTensor({B1, B2, B3, B4});

  Bond bd_sym_a = Bond(BD_KET, {{0, 2}, {3, 5}, {1, 6}, {4, 1}}, {4, 7, 2, 3});
  Bond bd_sym_b = Bond(BD_BRA, {{0, 2}, {3, 5}, {1, 6}, {4, 1}}, {4, 7, 2, 3});
  UniTensor BUT2 = UniTensor({bd_sym_a, bd_sym_b});

  Bond bd_sym_c =
    Bond(BD_KET, {{0, 2}, {1, 5}, {1, 6}, {0, 1}}, {4, 7, 2, 3}, {Symmetry::Zn(2), Symmetry::U1()});
  Bond bd_sym_d =
    Bond(BD_BRA, {{0, 2}, {1, 5}, {1, 6}, {0, 1}}, {4, 7, 2, 3}, {Symmetry::Zn(2), Symmetry::U1()});
  UniTensor BUT3 = UniTensor({bd_sym_c, bd_sym_d});

  Bond B1p = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {2, 1, 2});
  Bond B2p = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {4, 3, 4});
  Bond B3p = Bond(BD_IN, {Qs(-1), Qs(0), Qs(2)}, {1, 1, 1});
  Bond B4p = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {2, 1, 2});
  UniTensor BUT4 = UniTensor({B1p, B2p, B3p, B4p});
  UniTensor BUconjT4 = UniTensor({B1p, B2p, B3p, B4p});
  UniTensor BUtrT4 = UniTensor({B2p, B3p});

  Bond bd_sym_f =
    Bond(BD_KET, {{0, 2}, {1, 5}, {1, 6}, {0, 1}}, {4, 7, 2, 3}, {Symmetry::Zn(2), Symmetry::U1()});
  Bond bd_sym_g =
    Bond(BD_BRA, {{0, 2}, {1, 5}, {1, 6}, {0, 1}}, {4, 7, 2, 3}, {Symmetry::Zn(2), Symmetry::U1()});
  UniTensor BUT5 = UniTensor({bd_sym_f, bd_sym_g});

  BlockUniTensor BkUt;

  Tensor tzero345 = zeros({3, 4, 5});;
  UniTensor utzero345 = UniTensor(zeros(3 * 4 * 5)).reshape({3, 4, 5});

  Tensor t0;
  Tensor t1a;
  Tensor t1b;
  Tensor t2;

  Bond pBI = Bond(BD_IN,{Qs(0),Qs(1)},{2,3});
  Bond pBJ = Bond(BD_IN,{Qs(0),Qs(1)},{4,5});
  Bond pBK = Bond(BD_OUT,{Qs(0),Qs(1),Qs(2),Qs(3)},{6,7,8,9});

  Bond phy = Bond(BD_IN,{Qs(0),Qs(1)},{1,1});
  Bond aux = Bond(BD_IN,{Qs(1)},{1});

  Bond C1B1 = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {2, 3, 4});
  Bond C1B2 = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {2, 3, 4});
  Bond C2B1 = Bond(BD_IN, {Qs(-1), Qs(0), Qs(+1), Qs(+1), Qs(0)}, {2,2,2,2,1});
  Bond C2B2 = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(+1), Qs(+1), Qs(0)}, {2,2,2,2,1});
  Bond C3B1 = Bond(BD_IN, {Qs(0), Qs(1), Qs(0), Qs(1)}, {1,2,3,4});
  Bond C3B2 = Bond(BD_IN, {Qs(0), Qs(1), Qs(0), Qs(1)}, {1,2,3,4});
  Bond C3B3 = Bond(BD_OUT, {Qs(0), Qs(1), Qs(2)}, {1,2,3});

  UniTensor Spf = UniTensor({phy,phy.redirect(),aux},{1,2,3},1,Type.Float,Device.cpu,false);
  UniTensor Spd = UniTensor({phy,phy.redirect(),aux},{1,2,3},1,Type.Double,Device.cpu,false);
  UniTensor Spcf = UniTensor({phy,phy.redirect(),aux},{1,2,3},1,Type.ComplexFloat,Device.cpu,false);
  UniTensor Spcd = UniTensor({phy,phy.redirect(),aux},{1,2,3},1,Type.ComplexDouble,Device.cpu,false);

  UniTensor UT_pB = UniTensor({pBI,pBJ,pBK});
  UniTensor UT_pB_ans = UniTensor({pBI,pBJ,pBK});
  UniTensor UT_contract_L1 = UniTensor({C1B1, C2B2});
  UniTensor UT_contract_R1 = UniTensor({C1B1, C2B2});
  UniTensor UT_contract_ans1 = UniTensor({C1B1, C2B2});
  UniTensor UT_contract_L2 = UniTensor({C2B1, C2B2});
  UniTensor UT_contract_R2 = UniTensor({C2B1, C2B2});
  UniTensor UT_contract_ans2 = UniTensor({C2B1, C2B2});
  UniTensor UT_contract_L3 = UniTensor({C3B1, C3B2, C3B3});
  UniTensor UT_contract_R3 = UniTensor({C3B3.redirect(), C3B1.redirect(), C3B2.redirect()});
  UniTensor UT_contract_ans3 = UniTensor({C3B1, C3B2, C3B1.redirect(), C3B2.redirect()});

 protected:
  void SetUp() override {
    BUT4 = BUT4.Load("OriginalBUT.cytnx");
    BUconjT4 = BUconjT4.Load("BUconjT.cytnx");
    BUtrT4 = BUtrT4.Load("BUtrT.cytnx");

    t0 = Tensor::Load("put_block_t0.cytn");
    t1a = Tensor::Load("put_block_t1a.cytn");
    t1b = Tensor::Load("put_block_t1b.cytn");
    t2 = Tensor::Load("put_block_t2.cytn");

    UT_pB_ans = UT_pB_ans.Load("put_block_ans.cytnx");
    UT_contract_L1 = UT_contract_L1.Load("contract_L1.cytnx");
    UT_contract_R1 = UT_contract_R1.Load("contract_R1.cytnx");
    UT_contract_ans1 =  UT_contract_ans1.Load("contract_ans1.cytnx");
    UT_contract_L2 = UT_contract_L2.Load("contract_L2.cytnx");
    UT_contract_R2 = UT_contract_R2.Load("contract_R2.cytnx");
    UT_contract_ans2 =  UT_contract_ans2.Load("contract_ans2.cytnx");
    UT_contract_L3 = UT_contract_L3.Load("contract_L3.cytnx");
    UT_contract_R3 = UT_contract_R3.Load("contract_R3.cytnx");
    UT_contract_ans3 =  UT_contract_ans3.Load("contract_ans3.cytnx");
    
  }
  void TearDown() override {}
};

#endif
