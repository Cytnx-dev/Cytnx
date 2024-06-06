#include "Contract_test.h"

using namespace std;
using namespace cytnx;

TEST_F(ContractTest, gpu_Contract_denseUt_optimal_order) {
  UniTensor res = Contract({utdnA, utdnB, utdnC}, "", true);
  EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
}

TEST_F(ContractTest, gpu_Contract_denseUt_default_order) {
  UniTensor res = Contract({utdnA, utdnB, utdnC}, "", false);
  EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
}

TEST_F(ContractTest, gpu_Contract_denseUt_specified_order) {
  UniTensor res =
    Contract({utdnA.set_name("A"), utdnB.set_name("B"), utdnC.set_name("C")}, "(C,(A,B))", false);
  EXPECT_TRUE(AreNearlyEqTensor(res.get_block().contiguous(), utdnAns.get_block(), 1e-12));
}

TEST_F(ContractTest, gpu_Contract_denseUt_optimal_specified_order) {
  UniTensor res =
    Contract({utdnA.set_name("A"), utdnB.set_name("B"), utdnC.set_name("C")}, "(C,(A,B))", true);
  EXPECT_TRUE(AreNearlyEqTensor(res.get_block().contiguous(), utdnAns.get_block(), 1e-12));
}

// Deprecated Contracts
TEST_F(ContractTest, gpu_Contracts_denseUt_optimal_order) {
  UniTensor res = Contracts({utdnA, utdnB, utdnC}, "", true);
  EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
}

TEST_F(ContractTest, gpu_Contracts_denseUt_default_order) {
  UniTensor res = Contracts({utdnA, utdnB, utdnC}, "", false);
  EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
}

TEST_F(ContractTest, gpu_Contracts_denseUt_specified_order) {
  UniTensor res =
    Contracts({utdnA.set_name("A"), utdnB.set_name("B"), utdnC.set_name("C")}, "(C,(A,B))", false);
  EXPECT_TRUE(AreNearlyEqTensor(res.get_block().contiguous(), utdnAns.get_block(), 1e-12));
}

TEST_F(ContractTest, gpu_Contracts_denseUt_optimal_specified_order) {
  UniTensor res =
    Contracts({utdnA.set_name("A"), utdnB.set_name("B"), utdnC.set_name("C")}, "(C,(A,B))", true);
  EXPECT_TRUE(AreNearlyEqTensor(res.get_block().contiguous(), utdnAns.get_block(), 1e-12));
}
