#pragma once

#include <vector>
#include <unordered_map>
#include <set>
#include <bitset>
#include <cstdint>
#include <memory>

using IndexSet = std::bitset<128>;

class TreeNode {
 public:
  // Constructors
  explicit TreeNode(int index) : isLeaf_(true), tensorIndex_(index) {}
  TreeNode(std::unique_ptr<TreeNode> left, std::unique_ptr<TreeNode> right)
      : isLeaf_(false), left_(std::move(left)), right_(std::move(right)) {}

  // Accessors
  bool isLeaf() const { return isLeaf_; }
  int getTensorIndex() const { return tensorIndex_; }
  const TreeNode* getLeft() const { return left_.get(); }
  const TreeNode* getRight() const { return right_.get(); }

 private:
  bool isLeaf_;
  union {
    int tensorIndex_;  // for leaf nodes
    struct {  // for internal nodes
      std::unique_ptr<TreeNode> left_;
      std::unique_ptr<TreeNode> right_;
    };
  };
};

class ComponentData {
 public:
  void resize(size_t size) {
    costDict_.resize(size);
    treeDict_.resize(size);
    indexDict_.resize(size);
  }

  std::vector<std::unordered_map<IndexSet, int64_t>>& getCostDict() { return costDict_; }
  std::vector<std::unordered_map<IndexSet, std::unique_ptr<TreeNode>>>& getTreeDict() {
    return treeDict_;
  }
  std::vector<std::unordered_map<IndexSet, IndexSet>>& getIndexDict() { return indexDict_; }

 private:
  std::vector<std::unordered_map<IndexSet, int64_t>> costDict_;
  std::vector<std::unordered_map<IndexSet, std::unique_ptr<TreeNode>>> treeDict_;
  std::vector<std::unordered_map<IndexSet, IndexSet>> indexDict_;
};

class OptimalTreeResult {
 public:
  OptimalTreeResult(std::unique_ptr<TreeNode> tree, int64_t cost)
      : tree_(std::move(tree)), cost_(cost) {}

  const TreeNode* getTree() const { return tree_.get(); }
  int64_t getCost() const { return cost_; }

 private:
  std::unique_ptr<TreeNode> tree_;
  int64_t cost_;
};

class OptimalTreeSolver {
 public:
  static int64_t computeCost(const std::vector<int64_t>& allCosts, const IndexSet& ind1,
                             const IndexSet& ind2);

  static std::vector<std::vector<int>> findConnectedComponents(
    const std::vector<std::vector<bool>>& adjacencyMatrix);

  static OptimalTreeResult solve(const std::vector<std::vector<int>>& network,
                                 const std::unordered_map<int, int64_t>& optdata,
                                 bool verbose = false);
};
