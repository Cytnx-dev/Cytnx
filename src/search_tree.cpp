#include "search_tree.hpp"
#include <stack>

using namespace std;

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  // helper functions
  int64_t computeCost(const std::vector<int64_t>& allCosts, const IndexSet& ind1,
                      const IndexSet& ind2) {
    IndexSet result = ind1 | ind2;
    int64_t cost = 1;

    for (size_t i = 0; i < result.size(); ++i) {
      if (result[i]) {
        cost = cost * allCosts[i];
      }
    }
    return cost;
  }

  cytnx_float get_cost(const PseudoUniTensor& t1, const PseudoUniTensor& t2) {
    cytnx_float cost = 1;
    vector<cytnx_uint64> shape1 = t1.shape;
    vector<cytnx_uint64> shape2 = t2.shape;

    for (cytnx_uint64 i = 0; i < shape1.size(); i++) {
      cost *= shape1[i];
    }
    for (cytnx_uint64 i = 0; i < shape2.size(); i++) {
      cost *= shape2[i];
    }

    // get bond with common label:
    vector<string> common_lbl;
    vector<cytnx_uint64> comm_idx1, comm_idx2;
    vec_intersect_(common_lbl, t1.labels, t2.labels, comm_idx1, comm_idx2);

    for (cytnx_uint64 i = 0; i < comm_idx2.size(); i++) cost /= shape2[comm_idx2[i]];

    return cost + t1.cost + t2.cost;
  }

  PseudoUniTensor pContract(PseudoUniTensor& t1, PseudoUniTensor& t2) {
    PseudoUniTensor t3;
    t3.ID = t1.ID ^ t2.ID;
    t3.cost = get_cost(t1, t2);
    vector<cytnx_uint64> loc1, loc2;
    vector<string> comm_lbl;
    vec_intersect_(comm_lbl, t1.labels, t2.labels, loc1, loc2);
    t3.shape = vec_concatenate(vec_erase(t1.shape, loc1), vec_erase(t2.shape, loc2));
    t3.labels = vec_concatenate(vec_erase(t1.labels, loc1), vec_erase(t2.labels, loc2));
    t3.accu_str = "(" + t1.accu_str + "," + t2.accu_str + ")";
    return t3;
  }

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

  namespace OptimalTreeSolver {
    OptimalTreeResult solve(const std::vector<std::vector<int>>& network,
                            const std::unordered_map<int, int64_t>& dimensions, bool verbose) {
      std::vector<TreeNode*> nodes;
      for (size_t i = 0; i < network.size(); ++i) {
        nodes.push_back(new TreeNode(i));
      }

      while (nodes.size() > 1) {
        TreeNode* parent = new TreeNode();
        parent->setChildren(nodes[0], nodes[1]);
        nodes.erase(nodes.begin(), nodes.begin() + 2);
        nodes.insert(nodes.begin(), parent);
      }

      return OptimalTreeResult(nodes[0]);
    }
  }  // namespace OptimalTreeSolver

  void SearchTree::search_order() {
    this->reset_search_order();
    if (this->base_nodes.size() == 0) {
      cytnx_error_msg(true, "[ERROR][SearchTree] no base node exist.%s", "\n");
    }

    // Convert base_nodes to network format
    std::vector<std::vector<int>> network;
    std::unordered_map<int, int64_t> optdata;
    std::unordered_map<std::string, int> label_to_index;
    int next_index = 0;

    // First pass: collect all unique labels and assign indices
    for (const auto& node : base_nodes) {
      for (const auto& label : node.labels) {
        if (label_to_index.find(label) == label_to_index.end()) {
          label_to_index[label] = next_index++;
          // Use actual dimension from shape if available
          size_t pos =
            std::find(node.labels.begin(), node.labels.end(), label) - node.labels.begin();
          optdata[next_index - 1] = (pos < node.shape.size()) ? node.shape[pos] : 2;
        }
      }
    }

    // Second pass: convert nodes to network format
    for (size_t i = 0; i < base_nodes.size(); i++) {
      const auto& node = base_nodes[i];
      std::vector<int> tensor_indices;
      for (const auto& label : node.labels) {
        tensor_indices.push_back(label_to_index[label]);
      }
      network.push_back(tensor_indices);

      // Set ID for each base node
      base_nodes[i].ID = 1ULL << i;
    }

    // Run optimal tree solver
    auto result = OptimalTreeSolver::solve(network, optdata, false);

    // Convert the result back to PseudoUniTensor format
    std::function<std::unique_ptr<PseudoUniTensor>(const TreeNode*)> convert_tree;
    convert_tree = [&](const TreeNode* node) -> std::unique_ptr<PseudoUniTensor> {
      if (node->isLeaf()) {
        auto tensor = std::make_unique<PseudoUniTensor>();
        *tensor = base_nodes[node->getTensorIndex()];
        tensor->accu_str = std::to_string(node->getTensorIndex());
        return tensor;
      } else {
        auto left = convert_tree(node->getLeft());
        auto right = convert_tree(node->getRight());
        auto result = pContract(*left, *right);
        auto new_node = std::make_unique<PseudoUniTensor>();
        *new_node = std::move(result);
        new_node->accu_str = "(" + left->accu_str + "," + right->accu_str + ")";
        return new_node;
      }
    };

    // Convert and store the result
    root = convert_tree(result.getTree());
  }

}  // namespace cytnx
#endif
