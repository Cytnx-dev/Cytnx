#ifndef _H_contraction_tree_
#define _H_contraction_tree_

#include "Type.hpp"
#include "UniTensor.hpp"
#include "cytnx_error.hpp"
#include "utils/utils.hpp"
#include <vector>
#include <map>
#include <string>

namespace cytnx {
  /// @cond
  class Node {
   public:
    UniTensor utensor;  // don't worry about copy, because everything are references in cytnx!
    bool is_assigned;
    Node *left;
    Node *right;
    std::string name;
    Node *root;

    Node() : is_assigned(false), left(nullptr), right(nullptr), root(nullptr){};
    Node(const Node &rhs) {
      this->left = rhs.left;
      this->right = rhs.right;
      this->root = rhs.root;
      this->utensor = rhs.utensor;
      this->is_assigned = rhs.is_assigned;
    }
    Node &operator==(const Node &rhs) {
      this->left = rhs.left;
      this->right = rhs.right;
      this->root = rhs.root;
      this->utensor = rhs.utensor;
      this->is_assigned = rhs.is_assigned;
      return *this;
    }
    Node(Node *in_left, Node *in_right, const UniTensor &in_uten = UniTensor())
        : is_assigned(false), left(nullptr), right(nullptr), root(nullptr) {
      this->left = in_left;
      this->right = in_right;
      in_left->root = this;
      in_right->root = this;
      if (in_uten.uten_type() != UTenType.Void) this->utensor = in_uten;
    }
    void assign_utensor(const UniTensor &in_uten) {
      this->utensor = in_uten;
      this->is_assigned = true;
    }
    void clear_utensor() {
      this->is_assigned = false;
      this->utensor = UniTensor();
    }
  };

  class ContractionTree {
   public:
    std::vector<Node> nodes_container;  // this contains intermediate layer.
    std::vector<Node> base_nodes;  // this is the button layer.

    ContractionTree(){};
    ContractionTree(const ContractionTree &rhs) {
      this->nodes_container = rhs.nodes_container;
      this->base_nodes = rhs.base_nodes;
    }
    ContractionTree &operator==(const ContractionTree &rhs) {
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
    void reset_contraction_order() {
      nodes_container.clear();
      for (cytnx_uint64 i = 0; i < base_nodes.size(); i++) {
        base_nodes[i].root = nullptr;
      }
      // nodes_container.reserve(1024);
    }
    void reset_nodes() {
      // reset all nodes but keep the skeleton
      for (cytnx_uint64 i = 0; i < this->nodes_container.size(); i++) {
        this->nodes_container[i].clear_utensor();
      }
      for (cytnx_uint64 i = 0; i < this->base_nodes.size(); i++) {
        this->base_nodes[i].clear_utensor();
      }
    }
    void build_default_contraction_tree();
    void build_contraction_tree_by_tokens(const std::map<std::string, cytnx_uint64> &name2pos,
                                          const std::vector<std::string> &tokens);
  };
  /// @endcond
}  // namespace cytnx
#endif
