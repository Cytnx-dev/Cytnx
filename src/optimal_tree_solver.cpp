#include "optimal_tree_solver.hpp"
#include <limits>
#include <algorithm>

TreeNode::TreeNode(int tensor_idx) : tensor_index(tensor_idx), left(nullptr), right(nullptr) {}

bool TreeNode::isLeaf() const { return left == nullptr && right == nullptr; }

int TreeNode::getTensorIndex() const { return tensor_index; }

TreeNode* TreeNode::getLeft() const { return left; }

TreeNode* TreeNode::getRight() const { return right; }

void TreeNode::setChildren(TreeNode* l, TreeNode* r) {
  left = l;
  right = r;
}

OptimalTreeResult::OptimalTreeResult(TreeNode* t) : tree(t) {}

const TreeNode* OptimalTreeResult::getTree() const { return tree.get(); }

OptimalTreeResult OptimalTreeSolver::solve(const std::vector<std::vector<int>>& network,
                                           const std::unordered_map<int, int64_t>& dimensions,
                                           bool verbose) {
  // Create leaf nodes for each tensor
  std::vector<TreeNode*> nodes;
  for (size_t i = 0; i < network.size(); ++i) {
    nodes.push_back(new TreeNode(i));
  }

  // Simple greedy strategy: combine nearest neighbors
  while (nodes.size() > 1) {
    TreeNode* parent = new TreeNode();
    parent->setChildren(nodes[0], nodes[1]);
    nodes.erase(nodes.begin(), nodes.begin() + 2);
    nodes.insert(nodes.begin(), parent);
  }

  return OptimalTreeResult(nodes[0]);
}
