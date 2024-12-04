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
#include "UniTensor.hpp"

namespace cytnx {

  using IndexSet = std::bitset<128>;

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

    // Constructors
    explicit PseudoUniTensor(cytnx_uint64 index = 0)
        : isLeaf(true), tensorIndex(index), is_assigned(false), cost(0), ID(0) {}

    PseudoUniTensor(std::unique_ptr<PseudoUniTensor> l, std::unique_ptr<PseudoUniTensor> r)
        : isLeaf(false), left(std::move(l)), right(std::move(r)), cost(0), ID(0) {}

    // Copy and move constructors and assignment operators
    PseudoUniTensor() = default;  // Add default constructor
    PseudoUniTensor(const PseudoUniTensor& rhs);
    PseudoUniTensor(PseudoUniTensor&& rhs) noexcept;
    PseudoUniTensor& operator=(const PseudoUniTensor& rhs);
    PseudoUniTensor& operator=(PseudoUniTensor&& rhs) noexcept;
    ~PseudoUniTensor();

    void from_utensor(const UniTensor& in_uten);
    void clear_utensor();
  };

  struct OptimalTreeResult {
    std::unique_ptr<PseudoUniTensor> tree;
    int64_t cost;
  };

  struct ComponentData {
    std::vector<std::unordered_map<IndexSet, int64_t>> costDict;
    std::vector<std::unordered_map<IndexSet, std::unique_ptr<PseudoUniTensor>>> treeDict;
    std::vector<std::unordered_map<IndexSet, IndexSet>> indexDict;

    void resize(size_t size) {
      costDict.resize(size);
      treeDict.resize(size);
      indexDict.resize(size);
    }
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

    void search_order() {
      // Convert base_nodes to network format
      std::vector<std::vector<int>> network;
      std::unordered_map<int, int64_t> optdata;

      // Convert your existing nodes to the format expected by optimaltree
      for (const auto& node : base_nodes) {
        std::vector<int> tensor_indices;
        // Convert labels to indices as needed
        // Fill optdata with costs
        // ... conversion logic here ...
        network.push_back(tensor_indices);
      }

      // Define the optimaltree function within search_order
      auto optimaltree = [](const std::vector<std::vector<int>>& network,
                            const std::unordered_map<int, int64_t>& optdata) -> OptimalTreeResult {
        // Implement the logic of optimaltree here
        // This is a placeholder implementation
        OptimalTreeResult result;
        result.tree = std::make_unique<PseudoUniTensor>();
        result.cost = 0;
        // ... actual optimaltree logic ...
        return result;
      };

      // Run optimaltree
      auto result = optimaltree(network, optdata);

      // Store the result
      root = std::move(result.tree);
    }
  };

}  // namespace cytnx

#endif  // CYTNX_SEARCH_TREE_H_
