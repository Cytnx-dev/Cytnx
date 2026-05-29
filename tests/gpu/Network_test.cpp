#include "Network_test.h"

// TEST_F(NetworkTest, gpu_Network_stringLbl) {
//   auto Hi = Network();
//   EXPECT_NO_THROW(Hi.FromString({"A: a,e", "B: a,c_,d_,h", "C: e,f_,g_,h", "TOUT:
//   c_,d_;f_,g_"})); EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {ut1, ut2, ut3}));
//   EXPECT_NO_THROW(Hi.Launch(false));
//   EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3}));
//   EXPECT_NO_THROW(Hi.Launch(false));
// }

// TEST_F(NetworkTest, gpu_Network_integerLbl) {
//   auto Hi = Network();
//   EXPECT_NO_THROW(Hi.FromString({"A: 1,2", "B: 1,3,4,7", "C: 2,5,6,7", "TOUT: 3,4;5,6"}));
//   EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {ut1, ut2, ut3}));
//   EXPECT_NO_THROW(Hi.Launch(false));
//   EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3}));
//   EXPECT_NO_THROW(Hi.Launch(false));
// }

TEST_F(NetworkTest, gpu_Network_dense_FromString) {
  auto net = Network();
  net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
}

TEST_F(NetworkTest, gpu_Network_dense_no_order) {
  auto net = Network();
  net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
  net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
  EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
}

TEST_F(NetworkTest, gpu_Network_dense_find_optimal) {
  auto net = Network();
  net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
  net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
  net.setOrder(true, "");
  EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
}

TEST_F(NetworkTest, gpu_Network_dense_order_line) {
  auto net = Network();
  net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "ORDER:(A,(B,C))", "TOUT: a,b;e"});
  net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
  EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
}
TEST_F(NetworkTest, gpu_Network_dense_specified_order) {
  auto net = Network();
  net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
  net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
  net.setOrder(false, "(A,(B,C))");
  EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
}
TEST_F(NetworkTest, gpu_Network_dense_reuse) {
  auto net = Network();
  net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
  net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
  net.setOrder(false, "(A,(B,C))");
  EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
  net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnC, utdnB});
  EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
}

// Helper: Copy tensors to CPU. Contract them directly with Contract, and permute the open legs into
// the requested TOUT order.
namespace {
  UniTensor BlockNetworkReferenceCPU(const UniTensor& A, const UniTensor& B, const UniTensor& C) {
    UniTensor a = A.to(Device.cpu).relabel({"a", "e"});
    UniTensor b = B.to(Device.cpu).relabel({"a", "c_", "d_", "h"});
    UniTensor c = C.to(Device.cpu).relabel({"e", "f_", "g_", "h"});
    UniTensor expected = Contract(Contract(b, c), a);
    expected.permute_({"c_", "d_", "f_", "g_"}, 2);
    expected.contiguous_();
    return expected;
  }
}  // namespace

// Block (symmetric) UniTensor network contraction on the GPU. Validate the GPU result against a CPU
// contraction of the same tensors.
TEST_F(NetworkTest, gpu_Network_block_no_order) {
  random::uniform_(bkut1, -1., 1., 1);
  random::uniform_(bkut2, -1., 1., 2);
  random::uniform_(bkut3, -1., 1., 3);

  auto net = Network();
  net.FromString({"A: a,e", "B: a,c_,d_,h", "C: e,f_,g_,h", "TOUT: c_,d_;f_,g_"});
  net.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3});
  UniTensor res = net.Launch();
  EXPECT_EQ(res.uten_type(), UTenType.Block);
  EXPECT_EQ(res.device(), Device.cuda);

  UniTensor res_cpu = res.to(Device.cpu);
  res_cpu.contiguous_();
  EXPECT_TRUE(AreNearlyEqUniTensor(res_cpu, BlockNetworkReferenceCPU(bkut1, bkut2, bkut3), 1e-8));
}

TEST_F(NetworkTest, gpu_Network_block_specified_order) {
  random::uniform_(bkut1, -1., 1., 1);
  random::uniform_(bkut2, -1., 1., 2);
  random::uniform_(bkut3, -1., 1., 3);

  auto net = Network();
  net.FromString(
    {"A: a,e", "B: a,c_,d_,h", "C: e,f_,g_,h", "ORDER:(A,(B,C))", "TOUT: c_,d_;f_,g_"});
  net.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3});
  UniTensor res = net.Launch();

  UniTensor res_cpu = res.to(Device.cpu);
  res_cpu.contiguous_();
  EXPECT_TRUE(AreNearlyEqUniTensor(res_cpu, BlockNetworkReferenceCPU(bkut1, bkut2, bkut3), 1e-8));
}

// setOrder(true, "") on a non-dense network
TEST_F(NetworkTest, gpu_Network_block_find_optimal) {
  random::uniform_(bkut1, -1., 1., 1);
  random::uniform_(bkut2, -1., 1., 2);
  random::uniform_(bkut3, -1., 1., 3);

  auto net = Network();
  net.FromString({"A: a,e", "B: a,c_,d_,h", "C: e,f_,g_,h", "TOUT: c_,d_;f_,g_"});
  net.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3});
  net.setOrder(true, "");
  UniTensor res = net.Launch();

  UniTensor res_cpu = res.to(Device.cpu);
  res_cpu.contiguous_();
  EXPECT_TRUE(AreNearlyEqUniTensor(res_cpu, BlockNetworkReferenceCPU(bkut1, bkut2, bkut3), 1e-8));
}
