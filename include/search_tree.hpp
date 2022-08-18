#ifndef _H_search_tree_
#define _H_search_tree_

#include "Type.hpp"
#include "UniTensor.hpp"
#include "cytnx_error.hpp"
#include "utils/utils.hpp"
#include <vector>
#include <map>
#include <string>

namespace cytnx {
  /// @cond
  class PsudoUniTensor {
   public:
    // UniTensor utensor; //don't worry about copy, because everything are references in cytnx!
    std::vector<cytnx_int64> labels;
    std::vector<cytnx_uint64> shape;
    bool is_assigned;
    PsudoUniTensor *left;
    PsudoUniTensor *right;
    PsudoUniTensor *root;
    cytnx_float cost;
    cytnx_uint64 ID;

    std::string accu_str;

    PsudoUniTensor()
        : is_assigned(false), left(nullptr), right(nullptr), root(nullptr), cost(0), ID(0){};
    PsudoUniTensor(const PsudoUniTensor &rhs) {
      this->left = rhs.left;
      this->right = rhs.right;
      this->root = rhs.root;
      this->labels = rhs.labels;
      this->shape = rhs.shape;
      this->is_assigned = rhs.is_assigned;
      this->cost = rhs.cost;
      this->accu_str = rhs.accu_str;
      this->ID = rhs.ID;
    }
    PsudoUniTensor &operator==(const PsudoUniTensor &rhs) {
      this->left = rhs.left;
      this->right = rhs.right;
      this->root = rhs.root;
      this->labels = rhs.labels;
      this->shape = rhs.shape;
      this->is_assigned = rhs.is_assigned;
      this->cost = rhs.cost;
      this->accu_str = rhs.accu_str;
      this->ID = rhs.ID;
      return *this;
    }
    void from_utensor(const UniTensor &in_uten) {
      this->labels = in_uten.labels();
      this->shape = in_uten.shape();
      this->is_assigned = true;
    }
    void clear_utensor() {
      this->is_assigned = false;
      this->labels.clear();
      this->shape.clear();
      this->ID = 0;
      this->cost = 0;
      this->accu_str = "";
    }
    void set_ID(const cytnx_int64 &ID) { this->ID = ID; }
  };

  class SearchTree {
   public:
    std::vector<std::vector<PsudoUniTensor>> nodes_container;
    // std::vector<PsudoUniTensor> nodes_container; // this contains intermediate layer.
    std::vector<PsudoUniTensor> base_nodes;  // this is the button layer.

    SearchTree(){};
    SearchTree(const SearchTree &rhs) {
      this->nodes_container = rhs.nodes_container;
      this->base_nodes = rhs.base_nodes;
    }
    SearchTree &operator==(const SearchTree &rhs) {
      this->nodes_container = rhs.nodes_container;
      this->base_nodes = rhs.base_nodes;
      return *this;
    }

    // clear all the elements in the whole tree.
    void clear() {
      nodes_container.clear();
      base_nodes.clear();
      // nodes_container.reserve(1024);
    }
    // clear all the intermediate layer, leave all the base_nodes intact.
    // and reset the root pointer on the base ondes
    void reset_search_order() { nodes_container.clear(); }
    void search_order();
  };
  /// @endcond
}  // namespace cytnx
#endif
