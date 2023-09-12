#include "Contracts_test.h"

using namespace std;
using namespace cytnx;

TEST_F(ContractsTest, gpu_Contracts_denseUt_optimal_order) {
  UniTensor res = Contracts({utdnA, utdnB, utdnC}, "", true);
  EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
}

TEST_F(ContractsTest, gpu_Contracts_denseUt_specified_order) {
  UniTensor res = Contracts({utdnA, utdnB, utdnC}, "(C,(A,B))", false);
  EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
}

TEST_F(ContractsTest, gpu_Contracts_denseUt_default_order) {
  UniTensor res = Contracts({utdnA, utdnB, utdnC}, "", false);
  EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
}