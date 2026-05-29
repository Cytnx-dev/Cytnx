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

  std::vector<std::string> network_def = {"A: a,b,c", "B: c,d", "C: d,e", "ORDER:(A,(B,C))",
                                          "TOUT: a,b;e"};

  net.FromString(network_def);
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

// Helper: Contract three tensors directly with Contract, and permute the open legs into the
// requested TOUT order.
static UniTensor BlockNetworkReference(const UniTensor& A, const UniTensor& B, const UniTensor& C) {
  UniTensor a = A.relabel({"a", "e"});
  UniTensor b = B.relabel({"a", "c_", "d_", "h"});
  UniTensor c = C.relabel({"e", "f_", "g_", "h"});
  UniTensor expected = Contract(Contract(b, c), a);
  expected.permute_({"c_", "d_", "f_", "g_"}, 2);
  expected.contiguous_();
  return expected;
}

// Block (symmetric) UniTensor network contraction. Validate traversal/relabel/permute against a
// direct Contract of the same tensors.
TEST_F(NetworkTest, Network_block_no_order) {
  random::uniform_(bkut1, -1., 1., 1);
  random::uniform_(bkut2, -1., 1., 2);
  random::uniform_(bkut3, -1., 1., 3);

  auto net = Network();
  net.FromString({"A: a,e", "B: a,c_,d_,h", "C: e,f_,g_,h", "TOUT: c_,d_;f_,g_"});
  net.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3});
  UniTensor res = net.Launch();
  res.contiguous_();

  EXPECT_EQ(res.uten_type(), UTenType.Block);
  EXPECT_TRUE(AreNearlyEqUniTensor(res, BlockNetworkReference(bkut1, bkut2, bkut3), 1e-8));
}

TEST_F(NetworkTest, Network_block_specified_order) {
  random::uniform_(bkut1, -1., 1., 1);
  random::uniform_(bkut2, -1., 1., 2);
  random::uniform_(bkut3, -1., 1., 3);

  auto net = Network();
  net.FromString(
    {"A: a,e", "B: a,c_,d_,h", "C: e,f_,g_,h", "ORDER:(A,(B,C))", "TOUT: c_,d_;f_,g_"});
  net.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3});
  UniTensor res = net.Launch();
  res.contiguous_();

  EXPECT_TRUE(AreNearlyEqUniTensor(res, BlockNetworkReference(bkut1, bkut2, bkut3), 1e-8));
}

// setOrder(true, "") computes the optimal order
TEST_F(NetworkTest, Network_block_find_optimal) {
  random::uniform_(bkut1, -1., 1., 1);
  random::uniform_(bkut2, -1., 1., 2);
  random::uniform_(bkut3, -1., 1., 3);

  auto net = Network();
  net.FromString({"A: a,e", "B: a,c_,d_,h", "C: e,f_,g_,h", "TOUT: c_,d_;f_,g_"});
  net.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3});
  net.setOrder(true, "");
  UniTensor res = net.Launch();
  res.contiguous_();

  EXPECT_TRUE(AreNearlyEqUniTensor(res, BlockNetworkReference(bkut1, bkut2, bkut3), 1e-8));
}

// Helper: build a fermionic UniTensor with mixed in/out legs on both row and column spaces and
// degeneracies (rowrank 2), filled with sequential values over its existing components.
inline UniTensor make_mixed_inout_fermionic() {
  Bond B5Li = Bond(BD_IN, {Qs(0), Qs(1)}, {2, 1}, {Symmetry::FermionParity()});
  Bond B5Lo = Bond(BD_OUT, {Qs(0), Qs(1)}, {1, 2}, {Symmetry::FermionParity()});
  Bond B5Ri = Bond(BD_IN, {Qs(0), Qs(1)}, {1, 2}, {Symmetry::FermionParity()});
  Bond B5Ro = Bond(BD_OUT, {Qs(0), Qs(1)}, {2, 1}, {Symmetry::FermionParity()});
  UniTensor M = UniTensor({B5Li, B5Lo, B5Ri, B5Ro}, {"li", "lo", "ri", "ro"});
  M.set_rowrank_(2);
  cytnx_double val = 1.0;
  auto sh = M.shape();
  for (cytnx_uint64 i = 0; i < sh[0]; i++)
    for (cytnx_uint64 j = 0; j < sh[1]; j++)
      for (cytnx_uint64 k = 0; k < sh[2]; k++)
        for (cytnx_uint64 l = 0; l < sh[3]; l++) {
          auto proxy = M.at({i, j, k, l});
          if (proxy.exists()) {
            proxy = val;
            val += 1.0;
          }
        }
  return M;
}

// Helper: build a permuted (consistent {1,0,3,2}) copy and assert it carries non-trivial sign
// flips.
inline UniTensor permute_with_signflips(const UniTensor& M) {
  UniTensor Mp = M.permute({1, 0, 3, 2}).contiguous();
  bool anyflip = false;
  for (auto f : Mp.signflip()) anyflip = anyflip || f;
  EXPECT_TRUE(anyflip);  // ensure the signflip negation path is actually exercised
  return Mp;
}

// BlockFermionic network contraction: Launch must agree with a direct Contract of the same
// tensors. One input is permuted so its blocks carry pending sign flips; comparison uses apply(),
// which resolves those flips into the physical tensor.
TEST_F(NetworkTest, Network_fermionic_matches_contract) {
  UniTensor A = permute_with_signflips(make_mixed_inout_fermionic());  // legs (lo,li,ro,ri)
  UniTensor B = make_mixed_inout_fermionic();  // legs (li,lo,ri,ro)
  A.relabel_({"alo", "al", "m2", "m1"});  // A.ro -> m2 (OUT), A.ri -> m1 (IN)
  B.relabel_({"m2", "m1", "br", "bro"});  // B.li -> m2 (IN), B.lo -> m1 (OUT)

  auto net = Network();
  net.FromString({"A: alo,al,m2,m1", "B: m2,m1,br,bro", "TOUT: al,alo;br,bro"});
  net.PutUniTensors({"A", "B"}, {A, B});
  UniTensor res = net.Launch();

  UniTensor ref = Contract(A, B);
  ref.permute_({"al", "alo", "br", "bro"}, 2);
  EXPECT_EQ(res.uten_type(), UTenType.BlockFermionic);
  EXPECT_TRUE((res.apply() - ref.apply()).Norm().item() < 1e-8);
}
