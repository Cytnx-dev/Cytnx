#ifndef CYTNX_CONTRACTION_TREE_H_
#define CYTNX_CONTRACTION_TREE_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "UniTensor.hpp"
#include "utils/utils.hpp"
#include <vector>
#include <map>
#include <string>
#include <memory>

#ifdef BACKEND_TORCH
#else
namespace cytnx {
  /// @cond
  class Node : public std::enable_shared_from_this<Node> {
   public:
    UniTensor utensor;  // don't worry about copy, because everything are references in cytnx!
    bool is_assigned;
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;
    std::weak_ptr<Node> root;  // Use weak_ptr to avoid circular references
    std::string name;

    Node() : is_assigned(false) {}
    
    Node(const Node& rhs) 
        : utensor(rhs.utensor),
          is_assigned(rhs.is_assigned),
          left(rhs.left),
          right(rhs.right),
          name(rhs.name) {
      if (auto r = rhs.root.lock()) {
        root = r;
      }
    }
    
    Node& operator=(const Node& rhs) {
      if (this != &rhs) {
        utensor = rhs.utensor;
        is_assigned = rhs.is_assigned;
        left = rhs.left;
        right = rhs.right;
        name = rhs.name;
        if (auto r = rhs.root.lock()) {
          root = r;
        }
      }
      return *this;
    }

    Node(std::shared_ptr<Node> in_left, std::shared_ptr<Node> in_right, 
         const UniTensor& in_uten = UniTensor())
        : is_assigned(false) {
      left = in_left;
      right = in_right;
      if (in_uten.uten_type() != UTenType.Void) {
        utensor = in_uten;
      }
      
      // Set root pointers using shared_from_this()
      if (left) left->root = shared_from_this();
      if (right) right->root = shared_from_this();
    }

    void assign_utensor(const UniTensor& in_uten) {
      utensor = in_uten;
      is_assigned = true;
    }

    void clear_utensor() {
      is_assigned = false;
      utensor = UniTensor();
    }
  };

  class ContractionTree {
   public:
    std::vector<std::shared_ptr<Node>> nodes_container;  // intermediate layer
    std::vector<std::shared_ptr<Node>> base_nodes;       // bottom layer

    ContractionTree() = default;
    ContractionTree(const ContractionTree&) = default;
    ContractionTree& operator=(const ContractionTree&) = default;

    void clear() {
      nodes_container.clear();
      base_nodes.clear();
    }

    void reset_contraction_order() {
      nodes_container.clear();
      for (auto& node : base_nodes) {
        node->root.reset();  // Dereference shared_ptr with ->
      }
    }

    void reset_nodes() {
      for (auto& node : nodes_container) {
        node->clear_utensor();
      }
      for (auto& node : base_nodes) {
        node->clear_utensor();
      }
    }

    void build_default_contraction_tree();
    void build_contraction_tree_by_tokens(
        const std::map<std::string, cytnx_uint64>& name2pos,
        const std::vector<std::string>& tokens);
  };
  /// @endcond
}  // namespace cytnx
#endif  // BACKEND_TORCH

#endif  // CYTNX_CONTRACTION_TREE_H_
