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
  auto result = tree.get_nodes().back()[0];

  // Verify final cost is optimal
  EXPECT_EQ(result->cost, 32);  // 2*3*4 + 2*2*4 = 24 + 16 = 40 flops, cost = 32

  // Verify contraction string format
  EXPECT_TRUE(result->accu_str == "((0,1),2)" || result->accu_str == "((1,2),0)");
}

TEST_F(SearchTreeTest, EmptyTree) {
  SearchTree tree;

  // Should throw error when searching empty tree
  EXPECT_THROW(tree.search_order(), std::runtime_error);
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
  EXPECT_THROW(tree.search_order(), std::runtime_error);
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
  auto result = tree.get_nodes().back()[0];

  EXPECT_EQ(result->cost, 120);  // 2*3*4*5 = 120
  EXPECT_EQ(result->accu_str, "(0,1)");
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
