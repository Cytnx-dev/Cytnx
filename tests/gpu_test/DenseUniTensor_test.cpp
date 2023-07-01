#include "../DenseUniTensor_test.h"

using namespace std;
using namespace cytnx;
using namespace std::complex_literals;

TEST_F(DenseUniTensorTest, contract1_cutn) {
    ut1.set_labels({"a","b","c","d"});
    ut2.set_labels({"a","aa","bb","cc"});
    ut1.to_(Device.cuda);
    ut2.to_(Device.cuda);
    EXPECT_TRUE(ut1.dtype() == Type.ComplexDouble);
    EXPECT_TRUE(ut2.dtype() == Type.ComplexDouble);
    UniTensor out = ut1.contract(ut2);
    auto outbk = out.get_block_();
    auto ansbk = contres1.get_block_();
    EXPECT_TRUE(outbk.equiv(ansbk));
}

TEST_F(DenseUniTensorTest, contract2_cutn) {
    ut1.set_labels({"a","b","c","d"});
    ut2.set_labels({"a","b","bb","cc"});
    ut1.to_(Device.cuda);
    ut2.to_(Device.cuda);
    EXPECT_TRUE(ut1.dtype() == Type.ComplexDouble);
    EXPECT_TRUE(ut2.dtype() == Type.ComplexDouble);
    UniTensor out = ut1.contract(ut2);
    auto outbk = out.get_block_();
    auto ansbk = contres2.get_block_();
    EXPECT_TRUE(outbk.equiv(ansbk));
}

TEST_F(DenseUniTensorTest, contract3_cutn) {
    ut1.set_labels({"a","b","c","d"});
    ut2.set_labels({"a","b","c","cc"});
    ut1.to_(Device.cuda);
    ut2.to_(Device.cuda);
    EXPECT_TRUE(ut1.dtype() == Type.ComplexDouble);
    EXPECT_TRUE(ut2.dtype() == Type.ComplexDouble);
    UniTensor out = ut1.contract(ut2);
    auto outbk = out.get_block_();
    auto ansbk = contres3.get_block_();
    EXPECT_TRUE(outbk.equiv(ansbk));
}