#pragma once

#include <vector>
#include <unordered_map>
#include <set>
#include <bitset>
#include <cstdint>
#include <any>
#include <memory>

// Forward declarations
using IndexSet = std::bitset<128>;

struct OptimalTreeResult {
  std::unique_ptr<TreeNode> tree;
  int64_t cost;
};

struct TreeNode {
  bool isLeaf;
  union {
    int tensorIndex;  // for leaf nodes
    struct {  // for internal nodes
      std::unique_ptr<TreeNode> left;
      std::unique_ptr<TreeNode> right;
    };
  };

  // Constructors
  explicit TreeNode(int index) : isLeaf(true), tensorIndex(index) {}
  TreeNode(std::unique_ptr<TreeNode> l, std::unique_ptr<TreeNode> r)
      : isLeaf(false), left(std::move(l)), right(std::move(r)) {}
};

struct ComponentData {
  std::vector<std::unordered_map<IndexSet, int64_t>> costDict;
  std::vector<std::unordered_map<IndexSet, std::unique_ptr<TreeNode>>> treeDict;
  std::vector<std::unordered_map<IndexSet, IndexSet>> indexDict;

  void resize(size_t size) {
    costDict.resize(size);
    treeDict.resize(size);
    indexDict.resize(size);
  }
};

// Helper functions
int64_t addCost(int64_t cost1, int64_t cost2);
int64_t mulCost(int64_t cost1, int64_t cost2);
int64_t computeCost(const std::vector<int64_t> &allCosts, const IndexSet &ind1,
                    const IndexSet &ind2);
std::vector<std::vector<int>> connectedComponents(
  const std::vector<std::vector<bool>> &adjacencyMatrix);

// Main function
OptimalTreeResult optimaltree(const std::vector<std::vector<int>> &network,
                              const std::unordered_map<int, int64_t> &optdata,
                              bool verbose = false);
