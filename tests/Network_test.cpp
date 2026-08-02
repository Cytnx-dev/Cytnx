#include "Network_test.h"

// TEST_F(NetworkTest, NetworkStringLbl) {
//   auto Hi = Network();
//   EXPECT_NO_THROW(Hi.FromString({"A: a,e", "B: a,c_,d_,h", "C: e,f_,g_,h", "TOUT:
//   c_,d_;f_,g_"})); EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {ut1, ut2, ut3}));
//   EXPECT_NO_THROW(Hi.Launch(false));
//   EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3}));
//   EXPECT_NO_THROW(Hi.Launch(false));
// }

// TEST_F(NetworkTest, NetworkIntegerLbl) {
//   auto Hi = Network();
//   EXPECT_NO_THROW(Hi.FromString({"A: 1,2", "B: 1,3,4,7", "C: 2,5,6,7", "TOUT: 3,4;5,6"}));
//   EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {ut1, ut2, ut3}));
//   EXPECT_NO_THROW(Hi.Launch(false));
//   EXPECT_NO_THROW(Hi.PutUniTensors({"A", "B", "C"}, {bkut1, bkut2, bkut3}));
//   EXPECT_NO_THROW(Hi.Launch(false));
// }

namespace cytnx {
  namespace test {
    namespace {

      TEST_F(NetworkTest, NetworkDenseFromString) {
        auto net = Network();

        std::vector<std::string> network_def = {"A: a,b,c", "B: c,d", "C: d,e", "ORDER:(A,(B,C))",
                                                "TOUT: a,b;e"};

        net.FromString(network_def);
      }

      TEST_F(NetworkTest, NetworkOrderAcceptsBinaryGrammar) {
        struct OrderCase {
          std::string order;
          std::string expected_root;
        };
        const std::vector<OrderCase> order_cases = {
          {"(A,B),(C,D)", "((A,B),(C,D))"},
          {"((A,B),(C,D))", "((A,B),(C,D))"},
          {"(((A,B),C),D)", "(((A,B),C),D)"},
          {"(A,(B,(C,D)))", "(A,(B,(C,D)))"},
          {" ( A , B ) , ( C , D ) ", "((A,B),(C,D))"},
        };

        for (const OrderCase &order_case : order_cases) {
          SCOPED_TRACE(order_case.order);
          auto net = Network();
          net.FromString(
            {"A: a,b", "B: b,c", "C: c,d", "D: d,e", "TOUT: a;e", "ORDER: " + order_case.order});

          ASSERT_EQ(net._impl->CtTree.nodes_container.size(), 3);
          EXPECT_EQ(net._impl->CtTree.nodes_container.back()->name, order_case.expected_root);
        }

        for (const std::string &order : {"(A,B)", "A,B"}) {
          SCOPED_TRACE(order);
          auto net = Network();
          net.FromString({"A: a,b", "B: b,c", "TOUT: a;c", "ORDER: " + order});
          ASSERT_EQ(net._impl->CtTree.nodes_container.size(), 1);
          EXPECT_EQ(net._impl->CtTree.nodes_container.back()->name, "(A,B)");
        }

        auto named_net = Network();
        named_net.FromString({"A1: a,b", "B_Conj: b,c", "C2: c,d", "D_3: d,e", "TOUT: a;e",
                              "ORDER: (A1,B_Conj),(C2,D_3)"});
        ASSERT_EQ(named_net._impl->CtTree.nodes_container.size(), 3);
        EXPECT_EQ(named_net._impl->CtTree.nodes_container.back()->name, "((A1,B_Conj),(C2,D_3))");
      }

      TEST_F(NetworkTest, NetworkOrderRejectsMalformedExpressions) {
        struct MalformedOrderCase {
          std::string order;
          std::size_t expected_column;
        };
        const std::vector<MalformedOrderCase> malformed_orders = {
          {"A,B,C,D", 4},    {"(A,B,C,D)", 5},    {"((A,B)),(C,D)", 7}, {"(A,B,)", 5},
          {"),A,B,(", 1},    {"()(),A,B,C,D", 2}, {"(A,(B,C,D))", 8},   {"A,,B", 3},
          {"(A,B)(C,D)", 6}, {"(A,(B,C)", 9},     {"(A,B))", 6},
        };

        for (const MalformedOrderCase &order_case : malformed_orders) {
          SCOPED_TRACE(order_case.order);
          auto net = Network();
          try {
            net.FromString(
              {"A: a,b", "B: b,c", "C: c,d", "D: d,e", "TOUT: a;e", "ORDER: " + order_case.order});
            FAIL() << "accepted malformed ORDER expression: " << order_case.order;
          } catch (const error &exception) {
            const std::string message = exception.what();
            EXPECT_NE(message.find("[ERROR][ORDER]"), std::string::npos);
            EXPECT_NE(message.find("column:" + std::to_string(order_case.expected_column)),
                      std::string::npos);
          }
        }
      }

      TEST_F(NetworkTest, NetworkOrderRejectsSingleTensorAndInvalidCharacters) {
        struct InvalidOrderCase {
          std::string order;
          std::size_t expected_column;
          std::string expected_message;
        };
        const std::vector<InvalidOrderCase> invalid_orders = {
          {"A", 2, "expected at least two tensor names"},
          {"A;B", 2, "found a character that is not allowed in an ORDER expression"},
        };

        for (const InvalidOrderCase &order_case : invalid_orders) {
          SCOPED_TRACE(order_case.order);
          auto net = Network();
          try {
            net.FromString(
              {"A: a,b", "B: b,c", "C: c,d", "D: d,e", "TOUT: a;e", "ORDER: " + order_case.order});
            FAIL() << "accepted invalid ORDER expression: " << order_case.order;
          } catch (const error &exception) {
            const std::string message = exception.what();
            EXPECT_NE(message.find("[ERROR][ORDER]"), std::string::npos);
            EXPECT_NE(message.find("column:" + std::to_string(order_case.expected_column)),
                      std::string::npos);
            EXPECT_NE(message.find(order_case.expected_message), std::string::npos);
          }
        }
      }

      TEST_F(NetworkTest, NetworkSetOrderRequiresEveryTensorExactlyOnce) {
        struct OrderErrorCase {
          std::string order;
          std::string expected_message;
        };
        const std::vector<OrderErrorCase> order_cases = {
          {"(A,(B,(C,C)))", "duplicate tensor name: C"},
          {"(A,(B,(C,E)))", "undefined tensor name: E"},
          {"(A,(B,C))", "every tensor exactly once"},
        };

        for (const OrderErrorCase &order_case : order_cases) {
          SCOPED_TRACE(order_case.order);
          auto net = Network();
          net.FromString({"A: a,b", "B: b,c", "C: c,d", "D: d,e", "TOUT: a;e"});
          try {
            net.setOrder(false, order_case.order);
            FAIL() << "accepted incomplete ORDER expression: " << order_case.order;
          } catch (const error &exception) {
            EXPECT_NE(std::string(exception.what()).find(order_case.expected_message),
                      std::string::npos);
          }
        }
      }

      TEST_F(NetworkTest, NetworkDenseNoOrder) {
        auto net = Network();
        net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
        net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
        EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
      }

      TEST_F(NetworkTest, NetworkDenseFindOptimal) {
        auto net = Network();
        net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
        net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
        net.setOrder(true, "");
        EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
      }

      TEST_F(NetworkTest, NetworkDenseOrderLine) {
        auto net = Network();
        net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "ORDER:(A,(B,C))", "TOUT: a,b;e"});
        net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
        EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
      }

      TEST_F(NetworkTest, NetworkDenseSpecifiedOrder) {
        auto net = Network();
        net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
        net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
        net.setOrder(false, "(A,(B,C))");
        EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
      }

      TEST_F(NetworkTest, NetworkDenseReuse) {
        auto net = Network();
        net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
        net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
        net.setOrder(false, "(A,(B,C))");
        EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
        // EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
        net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnC, utdnB});
        EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
      }

      TEST_F(NetworkTest, NetworkDenseReuse2) {
        auto net = Network();
        net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b;e"});
        net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});

        EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
        EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
        EXPECT_TRUE(AreNearlyEqTensor(net.Launch().get_block(), utdnAns.get_block(), 1e-12));
      }

      TEST_F(NetworkTest, NetworkDenseTOUTNoColon) {
        auto net = Network();
        net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "TOUT: a,b,e"});
        net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
        auto res = net.Launch();
        EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
        EXPECT_EQ(res.rowrank(), 1);
      }

      // Helper: Contract three tensors directly with Contract, and permute the open legs into the
      // requested TOUT order.
      static UniTensor BlockNetworkReference(const UniTensor &A, const UniTensor &B,
                                             const UniTensor &C) {
        UniTensor a = A.relabel({"a", "e"});
        UniTensor b = B.relabel({"a", "c_", "d_", "h"});
        UniTensor c = C.relabel({"e", "f_", "g_", "h"});
        UniTensor expected = Contract(Contract(b, c), a);
        expected.permute_({"c_", "d_", "f_", "g_"}, 2);
        expected.contiguous_();
        return expected;
      }

      // Block (symmetric) UniTensor network contraction. Validate traversal/relabel/permute against
      // a direct Contract of the same tensors.
      TEST_F(NetworkTest, NetworkBlockNoOrder) {
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

      TEST_F(NetworkTest, NetworkBlockSpecifiedOrder) {
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
      TEST_F(NetworkTest, NetworkBlockFindOptimal) {
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

      // Helper: build a fermionic UniTensor with mixed in/out legs on both row and column spaces
      // and degeneracies (rowrank 2), filled with sequential values over its existing components.
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
      inline UniTensor permute_with_signflips(const UniTensor &M) {
        UniTensor Mp = M.permute({1, 0, 3, 2}).contiguous();
        bool anyflip = false;
        for (auto f : Mp.signflip()) anyflip = anyflip || f;
        EXPECT_TRUE(anyflip);  // ensure the signflip negation path is actually exercised
        return Mp;
      }

      // BlockFermionic network contraction: Launch must agree with a direct Contract of the same
      // tensors. One input is permuted so its blocks carry pending sign flips; comparison uses
      // apply(), which resolves those flips into the physical tensor.
      TEST_F(NetworkTest, NetworkFermionicMatchesContract) {
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

      TEST_F(NetworkTest, NetworkStaticContractDefaultOrder) {
        UniTensor A = utdnA.relabel({"a", "b", "c"});
        UniTensor B = utdnB.relabel({"c", "d"});
        UniTensor C = utdnC.relabel({"d", "e"});
        UniTensor res = Network::Contract({A, B, C}, "a,b;e").Launch();
        EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));
      }

      TEST_F(NetworkTest, NetworkStaticContractSpecifiedOrder) {
        UniTensor A = utdnA.relabel({"a", "b", "c"});
        UniTensor B = utdnB.relabel({"c", "d"});
        UniTensor C = utdnC.relabel({"d", "e"});
        UniTensor res =
          Network::Contract({A, B, C}, "a,b;e", {"A", "B", "C"}, "(A,(B,C))").Launch();
        EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), utdnAns.get_block(), 1e-12));

        // The static builder should agree with the equivalent FromString + PutUniTensors network.
        auto net = Network();
        net.FromString({"A: a,b,c", "B: c,d", "C: d,e", "ORDER:(A,(B,C))", "TOUT: a,b;e"});
        net.PutUniTensors({"A", "B", "C"}, {utdnA, utdnB, utdnC});
        UniTensor res_net = net.Launch();

        EXPECT_TRUE(AreNearlyEqTensor(res.get_block(), res_net.get_block(), 1e-12));
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
