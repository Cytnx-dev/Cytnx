#include "Network_test.h"

TEST_F(NetworkTest, gpu_Network_stringLbl) {
  auto Hi = Network();
  EXPECT_NO_THROW(Hi.FromString({"A: a,e", "B: a,c_,d_,h", "C: e,f_,g_,h", "TOUT: c_,d_;f_,g_"}));
  EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {ut1, ut2, ut3}));
  EXPECT_NO_THROW(Hi.Launch(false));
  EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3}));
  EXPECT_NO_THROW(Hi.Launch(false));
}

TEST_F(NetworkTest, gpu_Network_integerLbl) {
  auto Hi = Network();
  EXPECT_NO_THROW(Hi.FromString({"A: 1,2", "B: 1,3,4,7", "C: 2,5,6,7", "TOUT: 3,4;5,6"}));
  EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {ut1, ut2, ut3}));
  EXPECT_NO_THROW(Hi.Launch(false));
  EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3}));
  EXPECT_NO_THROW(Hi.Launch(false));
}
