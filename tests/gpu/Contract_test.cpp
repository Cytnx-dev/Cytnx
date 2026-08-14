#include "Contract_test.h"

namespace cytnx {
  namespace gpu_test {
    namespace {

      TEST_F(ContractTest, GpuContractDenseUtOptimalOrder) {
        UniTensor res = Contract({utdnA, utdnB, utdnC}, "", true);
        EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
      }

      TEST_F(ContractTest, GpuContractDenseUtDefaultOrder) {
        UniTensor res = Contract({utdnA, utdnB, utdnC}, "", false);
        EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
      }

      TEST_F(ContractTest, GpuContractDenseUtSpecifiedOrder) {
        UniTensor res = Contract({utdnA.set_name("A"), utdnB.set_name("B"), utdnC.set_name("C")},
                                 "(C,(A,B))", false);
        EXPECT_TRUE(AreNearlyEqTensor(res.get_block().contiguous(), utdnAns.get_block(), 1e-12));
      }

      TEST_F(ContractTest, GpuContractDenseUtOptimalSpecifiedOrder) {
        UniTensor res = Contract({utdnA.set_name("A"), utdnB.set_name("B"), utdnC.set_name("C")},
                                 "(C,(A,B))", true);
        EXPECT_TRUE(AreNearlyEqTensor(res.get_block().contiguous(), utdnAns.get_block(), 1e-12));
      }

      // Deprecated Contracts
      TEST_F(ContractTest, GpuContractsDenseUtOptimalOrder) {
        UniTensor res = Contracts({utdnA, utdnB, utdnC}, "", true);
        EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
      }

      TEST_F(ContractTest, GpuContractsDenseUtDefaultOrder) {
        UniTensor res = Contracts({utdnA, utdnB, utdnC}, "", false);
        EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
      }

      TEST_F(ContractTest, GpuContractsDenseUtSpecifiedOrder) {
        UniTensor res = Contracts({utdnA.set_name("A"), utdnB.set_name("B"), utdnC.set_name("C")},
                                  "(C,(A,B))", false);
        EXPECT_TRUE(AreNearlyEqTensor(res.get_block().contiguous(), utdnAns.get_block(), 1e-12));
      }

      TEST_F(ContractTest, GpuContractsDenseUtOptimalSpecifiedOrder) {
        UniTensor res = Contracts({utdnA.set_name("A"), utdnB.set_name("B"), utdnC.set_name("C")},
                                  "(C,(A,B))", true);
        EXPECT_TRUE(AreNearlyEqTensor(res.get_block().contiguous(), utdnAns.get_block(), 1e-12));
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
