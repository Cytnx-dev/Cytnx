#include "cytnx.hpp"
#include "search_tree.hpp"
#include <gtest/gtest.h>

using namespace cytnx;
using namespace std;

class SearchTreeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(SearchTreeTest, BasicSearchOrder) {
  // Create a simple network of 3 tensors
  vector<PseudoUniTensor> tensors;

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
  vector<PseudoUniTensor> tensors;

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
  cout << result->accu_str << endl;
  EXPECT_EQ(result->accu_str, "((2,3),(0,1))");
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
