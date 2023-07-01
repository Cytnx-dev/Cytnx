#include "../BlockUniTensor_test.h"

TEST_F(BlockUniTensorTest, contract1_cutn) {
    // two sparse matrix

    UT_contract_L1.set_labels({'a','b'});
    UT_contract_R1.set_labels({'b','c'});
    UT_contract_L1.to_(Device.cuda);
    UT_contract_R1.to_(Device.cuda);

    EXPECT_TRUE(UT_contract_L1.dtype() == Type.Double);
    EXPECT_TRUE(UT_contract_R1.dtype() == Type.Double);
    UniTensor out = UT_contract_L1.contract(UT_contract_R1);
    auto outbks = out.get_blocks();
    auto ansbks = UT_contract_ans1.get_blocks();
    for(int i = 0; i < ansbks.size(); i++)
        EXPECT_EQ(outbks[i].equiv(ansbks[i]), true);
}

TEST_F(BlockUniTensorTest, contract2_cutn) {
    // two sparse matrix with degeneracy

    UT_contract_L2.set_labels({'a','b'});
    UT_contract_R2.set_labels({'b','c'});
    UT_contract_L2.to_(Device.cuda);
    UT_contract_R2.to_(Device.cuda);
    EXPECT_TRUE(UT_contract_L2.dtype() == Type.Double);
    EXPECT_TRUE(UT_contract_R2.dtype() == Type.Double);
    UniTensor out = UT_contract_L2.contract(UT_contract_R2);
    auto outbks = out.get_blocks();
    auto ansbks = UT_contract_ans2.get_blocks();
    for(int i = 0; i < ansbks.size(); i++)
        EXPECT_EQ(outbks[i].equiv(ansbks[i]), true);
}

TEST_F(BlockUniTensorTest, contract3_cutn) {
    //// two 3 legs tensor

    UT_contract_L3.set_labels({'a','b','c'});
    UT_contract_R3.set_labels({'c','d','e'});
    UT_contract_L3.to_(Device.cuda);
    UT_contract_R3.to_(Device.cuda);
    // EXPECT_TRUE(UT_contract_L3.dtype() == Type.Double);
    // EXPECT_TRUE(UT_contract_R3.dtype() == Type.Double);
    UniTensor out = UT_contract_L3.contract(UT_contract_R3);
    auto outbks = out.get_blocks();
    auto ansbks = UT_contract_ans3.get_blocks();
    for(int i = 0; i < ansbks.size(); i++)
        EXPECT_EQ(outbks[i].equiv(ansbks[i]), true);
}