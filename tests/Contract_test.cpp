#include "Contract_test.h"
namespace cytnx {
  namespace {
    using test::ContractTest;

    TEST_F(ContractTest, ContractDenseUtOptimalOrder) {
      UniTensor res = Contract({utdnA, utdnB, utdnC}, "", true);
      EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
    }

    TEST_F(ContractTest, ContractDenseUtDefaultOrder) {
      UniTensor res = Contract({utdnA, utdnB, utdnC}, "", false);
      EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
    }

    TEST_F(ContractTest, ContractDenseUtSpecifiedOrder) {
      UniTensor res = Contract({utdnA.set_name("A"), utdnB.set_name("B"), utdnC.set_name("C")},
                               "(C,(A,B))", false);
      EXPECT_TRUE(AreNearlyEqTensor(res.get_block().contiguous(), utdnAns.get_block(), 1e-12));
    }

    TEST_F(ContractTest, ContractDenseUtOptimalSpecifiedOrder) {
      UniTensor res = Contract({utdnA.set_name("A"), utdnB.set_name("B"), utdnC.set_name("C")},
                               "(C,(A,B))", true);
      EXPECT_TRUE(AreNearlyEqTensor(res.get_block().contiguous(), utdnAns.get_block(), 1e-12));
    }

    // Deprecated Contracts
    TEST_F(ContractTest, ContractsDenseUtOptimalOrder) {
      UniTensor res = Contracts({utdnA, utdnB, utdnC}, "", true);
      EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
    }

    TEST_F(ContractTest, ContractsDenseUtDefaultOrder) {
      UniTensor res = Contracts({utdnA, utdnB, utdnC}, "", false);
      EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
    }

    TEST_F(ContractTest, ContractsDenseUtSpecifiedOrder) {
      UniTensor res = Contracts({utdnA.set_name("A"), utdnB.set_name("B"), utdnC.set_name("C")},
                                "(C,(A,B))", false);
      EXPECT_TRUE(AreNearlyEqTensor(res.get_block().contiguous(), utdnAns.get_block(), 1e-12));
    }

    TEST_F(ContractTest, ContractsDenseUtOptimalSpecifiedOrder) {
      UniTensor res = Contracts({utdnA.set_name("A"), utdnB.set_name("B"), utdnC.set_name("C")},
                                "(C,(A,B))", true);
      EXPECT_TRUE(AreNearlyEqTensor(res.get_block().contiguous(), utdnAns.get_block(), 1e-12));
    }
  }  // namespace
}  // namespace cytnx
