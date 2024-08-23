#include "BlockFermionicUniTensor_test.h"

TEST_F(BlockFermionicUniTensorTest, VectorContract) {
  // 1+2*2-3*3-4*4-5*5-6*6+7*7+8*8 = 32
  EXPECT_EQ(abs(BFUT1.contract(BFUT2).item() - 32) < 1e-12, true);
}

TEST_F(BlockFermionicUniTensorTest, SimpleTensorContract) {
  // 1+2*2-3*3-4*4-5*5-6*6+7*7+8*8 = 32
  EXPECT_EQ(abs(BFUT3.contract(BFUT2).at({0,0}) - 32) < 1e-13, true);
}

TEST_F(BlockFermionicUniTensorTest, SaveLoad) {
  BFUT1.Save("BFUT1");
  UniTensor BFUTloaded = BFUTloaded.Load("BFUT1.cytnx");
  EXPECT_EQ(AreEqUniTensor(BFUT1,BFUTloaded), true);
}
