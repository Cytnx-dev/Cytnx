#ifndef _H_search_tree_
#define _H_search_tree_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "utils/utils.hpp"
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  /// @cond
  class PseudoUniTensor : public std::enable_shared_from_this<PseudoUniTensor> {
   public:
    std::vector<std::string> labels;
    std::vector<cytnx_uint64> shape;
    bool is_assigned;
    std::shared_ptr<PseudoUniTensor> left;
    std::shared_ptr<PseudoUniTensor> right;
    std::shared_ptr<PseudoUniTensor> root;
    cytnx_float cost;
    cytnx_uint64 ID;
    std::string accu_str;

    PseudoUniTensor()
        : is_assigned(false), left(nullptr), right(nullptr), root(nullptr), cost(0), ID(0) {}
    PseudoUniTensor(const PseudoUniTensor &rhs)
        : labels(rhs.labels),
          shape(rhs.shape),
          is_assigned(rhs.is_assigned),
          left(rhs.left),
          right(rhs.right),
          root(rhs.root),
          cost(rhs.cost),
          ID(rhs.ID),
          accu_str(rhs.accu_str) {}
    PseudoUniTensor &operator=(const PseudoUniTensor &rhs) {
      if (this != &rhs) {
        labels = rhs.labels;
        shape = rhs.shape;
        is_assigned = rhs.is_assigned;
        left = rhs.left;
        right = rhs.right;
        root = rhs.root;
        cost = rhs.cost;
        ID = rhs.ID;
        accu_str = rhs.accu_str;
      }
      return *this;
    }
    void from_utensor(const UniTensor &in_uten) {
      labels = in_uten.labels();
      shape = in_uten.shape();
      is_assigned = true;
    }
    void clear_utensor() {
      is_assigned = false;
      labels.clear();
      shape.clear();
      ID = 0;
      cost = 0;
      accu_str.clear();
    }
    void set_ID(const cytnx_int64 &ID) { this->ID = ID; }
  };

  class SearchTree {
   public:
    std::vector<std::vector<std::shared_ptr<PseudoUniTensor>>> nodes_container;
    std::vector<std::shared_ptr<PseudoUniTensor>> base_nodes;

    SearchTree() = default;
    SearchTree(const SearchTree &rhs)
        : nodes_container(rhs.nodes_container), base_nodes(rhs.base_nodes) {}
    SearchTree &operator=(const SearchTree &rhs) {
      if (this != &rhs) {
        nodes_container = rhs.nodes_container;
        base_nodes = rhs.base_nodes;
      }
      return *this;
    }

    void clear() {
      nodes_container.clear();
      base_nodes.clear();
    }
    void reset_search_order() {
      nodes_container.clear();
      for (auto &node : base_nodes) {
        node->root = nullptr;
      }
    }
    void search_order();
  };
  /// @endcond
}  // namespace cytnx
#endif

#endif
