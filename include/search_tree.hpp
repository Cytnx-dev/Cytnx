#ifndef CYTNX_SEARCH_TREE_H_
#define CYTNX_SEARCH_TREE_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "utils/utils.hpp"
#include <vector>
#include <map>
#include <string>
#include <bitset>
#include <memory>
#include <unordered_map>
#include "UniTensor.hpp"

namespace cytnx {

  using IndexSet = std::bitset<128>;

  class TreeNode {
   public:
    TreeNode(int tensor_idx = -1) : tensor_index(tensor_idx), left(nullptr), right(nullptr) {}
    bool isLeaf() const { return left == nullptr && right == nullptr; }
    int getTensorIndex() const { return tensor_index; }
    TreeNode* getLeft() const { return left; }
    TreeNode* getRight() const { return right; }
    void setChildren(TreeNode* l, TreeNode* r) {
      left = l;
      right = r;
    }

   private:
    int tensor_index;
    TreeNode* left;
    TreeNode* right;
  };

  class OptimalTreeResult {
   public:
    OptimalTreeResult(TreeNode* tree = nullptr) : tree(tree) {}
    const TreeNode* getTree() const { return tree.get(); }

   private:
    std::unique_ptr<TreeNode> tree;
  };

  namespace OptimalTreeSolver {
    OptimalTreeResult solve(const std::vector<std::vector<int>>& network,
                            const std::unordered_map<int, int64_t>& dimensions,
                            bool verbose = false);
  }

  class PseudoUniTensor {
   public:
    bool isLeaf;
    union {
      // Leaf node data (tensor info)
      struct {
        std::vector<std::string> labels;
        std::vector<cytnx_uint64> shape;
        bool is_assigned;
        cytnx_uint64 tensorIndex;  // Added for optimaltree compatibility
      };
      // Internal node data (tree structure)
      struct {
        std::unique_ptr<PseudoUniTensor> left;
        std::unique_ptr<PseudoUniTensor> right;
      };
    };

    cytnx_float cost;
    cytnx_uint64 ID;
    std::string accu_str;
    std::shared_ptr<PseudoUniTensor> root;  // For tracking the root during construction

    // Constructors
    explicit PseudoUniTensor(cytnx_uint64 index = 0)
        : isLeaf(true), tensorIndex(index), is_assigned(false), cost(0), ID(0) {}

    PseudoUniTensor(std::unique_ptr<PseudoUniTensor> l, std::unique_ptr<PseudoUniTensor> r)
        : isLeaf(false), left(std::move(l)), right(std::move(r)), cost(0), ID(0) {}

    // Copy and move constructors and assignment operators
    PseudoUniTensor(const PseudoUniTensor& rhs);
    PseudoUniTensor(PseudoUniTensor&& rhs) noexcept;
    PseudoUniTensor& operator=(const PseudoUniTensor& rhs);
    PseudoUniTensor& operator=(PseudoUniTensor&& rhs) noexcept;
    ~PseudoUniTensor();

    void from_utensor(const UniTensor& in_uten);
    void clear_utensor();
  };

  class SearchTree {
   public:
    std::unique_ptr<PseudoUniTensor> root;
    std::vector<PseudoUniTensor> base_nodes;

    SearchTree() = default;

    void clear() {
      root.reset();
      base_nodes.clear();
    }

    void reset_search_order() { root.reset(); }
    void search_order();
  };

  // Helper functions declarations
  cytnx_float get_cost(const PseudoUniTensor& t1, const PseudoUniTensor& t2);
  PseudoUniTensor pContract(PseudoUniTensor& t1, PseudoUniTensor& t2);

}  // namespace cytnx

#endif  // CYTNX_SEARCH_TREE_H_
