#include <algorithm>
#include <set>

#include <gtest/gtest.h>

#include "cytnx.hpp"
#include "search_tree.hpp"

namespace cytnx {
  namespace test {

    class SearchTreeTest : public ::testing::Test {
     protected:
      void SetUp() override {}
      void TearDown() override {}
    };

    TEST_F(SearchTreeTest, BasicSearchOrder) {
      // Create a simple network of 3 tensors
      std::vector<PseudoUniTensor> tensors;

      // Create tensor 1 with shape [2,3] and labels ["i","j"]
      PseudoUniTensor t1(0);
      t1.shape = {2, 3};
      t1.labels = {"i", "j"};
      t1.cost = 0;
      tensors.push_back(t1);

      // Create tensor 2 with shape [3,4] and labels ["j","k"]
      PseudoUniTensor t2(1);
      t2.shape = {3, 4};
      t2.labels = {"j", "k"};
      t2.cost = 0;
      tensors.push_back(t2);

      // Create tensor 3 with shape [4,2] and labels ["k","i"]
      PseudoUniTensor t3(2);
      t3.shape = {4, 2};
      t3.labels = {"k", "i"};
      t3.cost = 0;
      tensors.push_back(t3);

      // Create search tree
      SearchTree tree;
      tree.base_nodes = tensors;

      // Find optimal contraction order
      tree.search_order();

      // Get result node
      auto result = tree.get_root().back()[0];

      // Verify final cost is optimal
      EXPECT_EQ(result->cost, 32);  // 2*3*4 + 2*2*4 = 24 + 16 = 40 flops, cost = 32

      // Verify contraction string format
      EXPECT_EQ(result->accu_str, "(2,(0,1))");
    }

    TEST_F(SearchTreeTest, BasicSearchOrder2) {
      // Create a network of 4 tensors to test more complex contraction ordering
      std::vector<PseudoUniTensor> tensors;

      // Create tensor 1 with shape [2,10] and labels ["i","j"]
      // This will connect with tensor 2 through index j
      PseudoUniTensor t1(0);
      t1.shape = {2, 10};
      t1.labels = {"i", "j"};
      t1.cost = 0;
      tensors.push_back(t1);

      // Create tensor 2 with shape [10,4,8] and labels ["j","k","m"]
      // This connects with t1 through j, t3 through k, and t4 through m
      PseudoUniTensor t2(1);
      t2.shape = {10, 4, 8};
      t2.labels = {"j", "k", "m"};
      t2.cost = 0;
      tensors.push_back(t2);

      // Create tensor 3 with shape [4,2] and labels ["k","l"]
      // This connects with t2 through k and t4 through l
      PseudoUniTensor t3(2);
      t3.shape = {4, 2};
      t3.labels = {"k", "l"};
      t3.cost = 0;
      tensors.push_back(t3);

      // Create tensor 4 with shape [2,8,7] and labels ["l","m","n"]
      // This connects with t3 through l and t2 through m
      // The n index remains uncontracted as an external index
      PseudoUniTensor t4(3);
      t4.shape = {2, 8, 7};
      t4.labels = {"l", "m", "n"};
      t4.cost = 0;
      tensors.push_back(t4);

      // Create search tree and set base nodes
      SearchTree tree;
      tree.base_nodes = tensors;

      // Find optimal contraction order using search algorithm
      tree.search_order();

      // Get the final contracted result node
      auto result = tree.get_root().back()[0];

      // Verify the final contraction cost is optimal
      // The optimal sequence contracts (t1,t2) first, then t3, then t4
      EXPECT_EQ(result->cost, 1536);

      // Verify the contraction sequence string matches expected optimal order
      // Format is (tensor_id,(tensor_id,tensor_id)) showing order of pairwise contractions
      std::cout << result->accu_str << std::endl;
      EXPECT_EQ(result->accu_str, "((2,3),(0,1))");
    }

    /*=====test info=====
    describe:regression test for issue #853. A single connected component of 5
             tensors needs 4 sequential contractions to collapse to one node, so
             by the second contraction the pair-selection loop must look up
             adjacency for a merged node whose index is >= the original leaf
             count. Bounds-checking that lookup (e.g. adjacencyMatrix.at(i))
             instead of re-deriving adjacency from the merged node's current
             labels would turn the heap-buffer-overflow into a clean
             std::out_of_range, which still fails this test: the merged node's
             index is used for every remaining node, so search_order() would
             throw before result->labels/result->ID could be checked. Reproduces
             the exact node count (n=4 leaves reachable) and connectivity shape
             called out in the issue's ASan trace, extended by one node to also
             exercise a second out-of-range lookup.
    ====================*/
    TEST_F(SearchTreeTest, ChainNetworkNoOutOfBoundsOnMultiStepContraction) {
      // Chain topology a-T0-b-T1-c-T2-d-T3-e-T4-f: every adjacent pair shares
      // exactly one label, so the whole network is one connected component and
      // only "a" and "f" remain uncontracted in the final result.
      std::vector<PseudoUniTensor> tensors;

      PseudoUniTensor t0(0);
      t0.shape = {2, 3};
      t0.labels = {"a", "b"};
      t0.cost = 0;
      tensors.push_back(t0);

      PseudoUniTensor t1(1);
      t1.shape = {3, 4};
      t1.labels = {"b", "c"};
      t1.cost = 0;
      tensors.push_back(t1);

      PseudoUniTensor t2(2);
      t2.shape = {4, 5};
      t2.labels = {"c", "d"};
      t2.cost = 0;
      tensors.push_back(t2);

      PseudoUniTensor t3(3);
      t3.shape = {5, 6};
      t3.labels = {"d", "e"};
      t3.cost = 0;
      tensors.push_back(t3);

      PseudoUniTensor t4(4);
      t4.shape = {6, 7};
      t4.labels = {"e", "f"};
      t4.cost = 0;
      tensors.push_back(t4);

      SearchTree tree;
      tree.base_nodes = tensors;

      EXPECT_NO_THROW(tree.search_order());

      auto result = tree.get_root().back()[0];
      ASSERT_NE(result, nullptr);

      // All 5 leaves must be folded into the result (ID is the XOR, i.e. union,
      // of the leaves' distinct power-of-two IDs).
      EXPECT_EQ(result->ID, (1ULL << 0) | (1ULL << 1) | (1ULL << 2) | (1ULL << 3) | (1ULL << 4));

      // Only the two external labels survive contraction.
      std::vector<std::string> labels = result->labels;
      std::sort(labels.begin(), labels.end());
      EXPECT_EQ(labels, std::vector<std::string>({"a", "f"}));
    }

    TEST_F(SearchTreeTest, EmptyTree) {
      SearchTree tree;

      // Should throw error when searching empty tree
      EXPECT_THROW(tree.search_order(), std::logic_error);
    }

    TEST_F(SearchTreeTest, SingleNode) {
      SearchTree tree;

      // Add single node
      PseudoUniTensor t1(0);
      t1.shape = {2, 2};
      t1.labels = {"i", "i"};
      t1.cost = 0;
      tree.base_nodes.push_back(t1);

      // Should throw error - need at least 2 nodes
      EXPECT_THROW(tree.search_order(), std::logic_error);
    }

    /*=====test info=====
    describe:guards that a merged node's adjacency is visible from BOTH
             directions. When two nodes are contracted, the incremental
             adjacency scheme must record the merged node as a neighbour of
             every node it inherits an edge from -- not only store the merged
             node's own outgoing row. The merged node always has the largest
             index and is compared as the second operand of the pair-selection
             loop's adjacencyMatrix[i].test(j), so if only its own row were
             updated the edge (older_node -> merged_node) would never be seen
             and the greedy planner would skip a cheap contraction it should
             have taken.

             Topology (chain, one shared label per adjacent pair):
               t0{x,y} t1{y,z} t2{z,w} t3{w,v}, dims x=y=10, z=3, w=4, v=2.
             With correct adjacency the greedy order is (t2,t3) -> t1 -> t0 at
             total cost 284. If the merged (t2,t3) node's adjacency to t1 is
             missed, the planner instead contracts (t0,t1) and finishes at the
             higher cost 384, so asserting on the cost (and accu_str) distinguishes
             the two.
    ====================*/
    TEST_F(SearchTreeTest, MergedNodeAdjacencyIsSymmetric) {
      std::vector<PseudoUniTensor> tensors;

      PseudoUniTensor t0(0);
      t0.shape = {10, 10};
      t0.labels = {"x", "y"};
      t0.cost = 0;
      tensors.push_back(t0);

      PseudoUniTensor t1(1);
      t1.shape = {10, 3};
      t1.labels = {"y", "z"};
      t1.cost = 0;
      tensors.push_back(t1);

      PseudoUniTensor t2(2);
      t2.shape = {3, 4};
      t2.labels = {"z", "w"};
      t2.cost = 0;
      tensors.push_back(t2);

      PseudoUniTensor t3(3);
      t3.shape = {4, 2};
      t3.labels = {"w", "v"};
      t3.cost = 0;
      tensors.push_back(t3);

      SearchTree tree;
      tree.base_nodes = tensors;
      tree.search_order();
      auto result = tree.get_root().back()[0];
      ASSERT_NE(result, nullptr);

      // Cheapest greedy order keeps the merged (t2,t3) node adjacent to t1.
      EXPECT_EQ(result->cost, 284);
      EXPECT_EQ(result->accu_str, "(0,(1,(2,3)))");
    }

    TEST_F(SearchTreeTest, TooManyNodesThrows) {
      // The solver caps at 64 tensors: leaf IDs are 1ULL << leaf_index (64-bit)
      // and adjacency rows are std::bitset<128> (up to 2*64-1 = 127 nodes), so a
      // 65th tensor would shift/index past both. Expect a clean error, not UB.
      //
      // solve() rejects the input on size alone, before building any leaf, so the
      // dummy nodes' contents are irrelevant. Use a fixed in-range constructor
      // index: PseudoUniTensor sets ID to 1ULL << index, so passing 64 would be
      // undefined behavior at construction, before the guard runs.
      SearchTree tree;
      for (cytnx_uint64 i = 0; i < 65; ++i) {
        PseudoUniTensor t(0);
        t.shape = {2, 2};
        t.labels = {std::to_string(i), std::to_string(i + 1)};
        t.cost = 0;
        tree.base_nodes.push_back(t);
      }
      EXPECT_THROW(tree.search_order(), std::logic_error);
    }

    TEST_F(SearchTreeTest, DisconnectedNetwork) {
      SearchTree tree;

      // Create two tensors with no common indices
      PseudoUniTensor t1(0);
      t1.shape = {2, 3};
      t1.labels = {"i", "j"};
      t1.cost = 0;
      tree.base_nodes.push_back(t1);

      PseudoUniTensor t2(1);
      t2.shape = {4, 5};
      t2.labels = {"k", "l"};
      t2.cost = 0;
      tree.base_nodes.push_back(t2);

      // Should still work but with higher cost due to direct product
      tree.search_order();
      auto result = tree.get_root().back()[0];

      EXPECT_EQ(result->cost, 120);  // 2*3*4*5 = 120
      EXPECT_EQ(result->accu_str, "(0,1)");
    }

    /*=====test info=====
    describe:regression test for issue #506. The 21-tensor iPEPS observable
             network below is the one whose optimal contraction order the CPU
             solver "could not finish in 1 day". It is a single connected
             component that needs 20 sequential contractions, so an
             O(nodes^2)-per-round rescan that recomputes every pair's cost each
             round spends O(nodes^3) get_cost calls -- each allocating -- and does
             not terminate in practical time. Guards that solve() completes and
             fully contracts the network (every shared bond is summed, the network
             file's TOUT is empty, so no label survives). The exact cost is not
             asserted: the greedy planner's tie-breaking is not part of the
             contract here, only that it terminates and collapses the whole
             network.

             Bonds are taken verbatim from iPEPS_observe.net; bond dimensions are
             chi = 8 for every leg except the physical d = 2 legs (O's four legs
             and the A/B tensors' last legs: labels 7, 8, 11, 12, 25, 30).
    ====================*/
    TEST_F(SearchTreeTest, IPEPSObservableNetworkContractsQuickly) {
      // Labels carrying the physical dimension d = 2; every other bond is chi = 8.
      const std::set<std::string> d2 = {"7", "8", "11", "12", "25", "30"};
      auto dim_of = [&](const std::string& label) -> cytnx_uint64 {
        return d2.count(label) ? 2 : 8;
      };

      const std::vector<std::vector<std::string>> net = {
        {"1", "2"},  // C1
        {"34", "35"},  // C2
        {"41", "42"},  // C3
        {"19", "16"},  // C4
        {"1", "20", "3", "4"},  // T1b
        {"20", "34", "21", "22"},  // T1a
        {"35", "38", "36", "37"},  // T2a
        {"38", "41", "39", "40"},  // T2b
        {"42", "33", "31", "32"},  // T3b
        {"33", "19", "17", "18"},  // T3a
        {"16", "13", "14", "15"},  // T4a
        {"13", "2", "5", "6"},  // T4b
        {"4", "24", "10", "6", "8"},  // A1
        {"3", "23", "9", "5", "7"},  // A1T
        {"22", "37", "27", "24", "25"},  // B1
        {"21", "36", "26", "23", "25"},  // B1T
        {"27", "40", "32", "29", "30"},  // A2
        {"26", "39", "31", "28", "30"},  // A2T
        {"10", "29", "18", "15", "12"},  // B2
        {"9", "28", "17", "14", "11"},  // B2T
        {"7", "8", "11", "12"},  // O
      };

      SearchTree tree;
      for (cytnx_uint64 i = 0; i < net.size(); ++i) {
        PseudoUniTensor t(i);
        t.labels = net[i];
        for (const std::string& label : net[i]) t.shape.push_back(dim_of(label));
        t.cost = 0;
        tree.base_nodes.push_back(t);
      }

      ASSERT_NO_THROW(tree.search_order());
      auto result = tree.get_root().back()[0];
      ASSERT_NE(result, nullptr);

      // Every one of the 21 leaves is folded into the result exactly once.
      EXPECT_EQ(result->ID, (1ULL << net.size()) - 1);
      // The network is fully contracted: no bond survives (TOUT is empty).
      EXPECT_TRUE(result->labels.empty());
    }
  }  // namespace test
}  // namespace cytnx
