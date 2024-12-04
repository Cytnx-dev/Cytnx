#ifndef _OPTIMAL_TREE_SOLVER_HPP_
#define _OPTIMAL_TREE_SOLVER_HPP_

#include <vector>
#include <unordered_map>
#include <memory>

class TreeNode {
 public:
  TreeNode(int tensor_idx = -1);
  bool isLeaf() const;
  int getTensorIndex() const;
  TreeNode* getLeft() const;
  TreeNode* getRight() const;
  void setChildren(TreeNode* left, TreeNode* right);

 private:
  int tensor_index;
  TreeNode* left;
  TreeNode* right;
};

class OptimalTreeResult {
 public:
  OptimalTreeResult(TreeNode* tree = nullptr);
  const TreeNode* getTree() const;

 private:
  std::unique_ptr<TreeNode> tree;
};

class OptimalTreeSolver {
 public:
  static OptimalTreeResult solve(const std::vector<std::vector<int>>& network,
                                 const std::unordered_map<int, int64_t>& dimensions,
                                 bool verbose = false);
};

#endif
