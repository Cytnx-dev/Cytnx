#include "Network_test.h"

// TEST_F(NetworkTest, Network_stringLbl) {
//   auto Hi = Network();
//   EXPECT_NO_THROW(Hi.FromString({"A: a,e", "B: a,c_,d_,h", "C: e,f_,g_,h", "TOUT:
//   c_,d_;f_,g_"})); EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {ut1, ut2, ut3}));
//   EXPECT_NO_THROW(Hi.Launch(false));
//   EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3}));
//   EXPECT_NO_THROW(Hi.Launch(false));
// }

// TEST_F(NetworkTest, Network_integerLbl) {
//   auto Hi = Network();
//   EXPECT_NO_THROW(Hi.FromString({"A: 1,2", "B: 1,3,4,7", "C: 2,5,6,7", "TOUT: 3,4;5,6"}));
//   EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {ut1, ut2, ut3}));
//   EXPECT_NO_THROW(Hi.Launch(false));
//   EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3}));
//   EXPECT_NO_THROW(Hi.Launch(false));
// }

TEST_F(NetworkTest, Network_dense_FromString) {
  auto net = Network();
  net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "ORDER:(A,(B,C))", "TOUT: a,b;e"});
}

TEST_F(NetworkTest, Network_dense_no_order) {
  auto net = Network();
  net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
  net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
  EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
}

TEST_F(NetworkTest, Network_dense_find_optimal) {
  auto net = Network();
  net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
  net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
  net.setOrder(true, "");
  EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
}

TEST_F(NetworkTest, Network_dense_order_line) {
  auto net = Network();
  net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "ORDER:(A,(B,C))", "TOUT: a,b;e"});
  net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
  EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
}

TEST_F(NetworkTest, Network_dense_specified_order) {
  auto net = Network();
  net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
  net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
  net.setOrder(false, "(A,(B,C))");
  EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
}

TEST_F(NetworkTest, Network_dense_reuse) {
  auto net = Network();
  net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
  net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
  net.setOrder(false, "(A,(B,C))");
  EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
  // EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
  net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnC, utdnB});
  EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
}

TEST_F(NetworkTest, Network_dense_reuse2) {
  auto net = Network();
  net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
  net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});

  EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
  EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
  EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
}

TEST_F(NetworkTest, Network_dense_TOUT_no_colon) {
  auto net = Network();
  net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b,e"});
  net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
  auto res = net.Launch();
  EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
  EXPECT_EQ(res.rowrank(), 1);
}
