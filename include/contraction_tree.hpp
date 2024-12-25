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
    UniTensor utensor;
    bool is_assigned;
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;
    std::weak_ptr<Node> root;
    std::string name;

    Node() : is_assigned(false) {}

    Node(const Node& rhs)
        : utensor(rhs.utensor),
          is_assigned(rhs.is_assigned),
          left(rhs.left),
          right(rhs.right),
          name(rhs.name) {
      // Only copy root if it exists
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
        : is_assigned(false), left(in_left), right(in_right) {
      // Set name based on children
      if (left && right) {
        name = "(" + left->name + "," + right->name + ")";
      }

      if (in_uten.uten_type() != UTenType.Void) {
        utensor = in_uten;
      }
    }

    void set_root_ptrs() {
      try {
        auto self = shared_from_this();

        if (left) {
          left->root = self;
          left->set_root_ptrs();
        }

        if (right) {
          right->root = self;
          right->set_root_ptrs();
        }
      } catch (const std::bad_weak_ptr& e) {
        std::cerr << "Failed to set root ptrs for node " << name << ": " << e.what() << std::endl;
        throw;
      }
    }

    void clear_utensor() {
      if (left) {
        left->clear_utensor();
        left->root.reset();
      }
      if (right) {
        right->clear_utensor();
        right->root.reset();
      }
      is_assigned = false;
      utensor = UniTensor();
    }

    void assign_utensor(const UniTensor& in_uten) {
      utensor = in_uten;
      is_assigned = true;
    }
  };

  class ContractionTree {
   public:
    std::vector<std::shared_ptr<Node>> nodes_container;  // intermediate layer
    std::vector<std::shared_ptr<Node>> base_nodes;  // bottom layer

    ContractionTree() = default;
    ContractionTree(const ContractionTree&) = default;
    ContractionTree& operator=(const ContractionTree&) = default;

    void clear() {
      nodes_container.clear();
      base_nodes.clear();
    }

    void reset_contraction_order() {
      // First clear all root pointers
      for (auto& node : base_nodes) {
        if (node) node->root.reset();
      }
      // Then clear the container
      nodes_container.clear();
    }

    void reset_nodes() {
      // Clear from root down if we have nodes
      if (!nodes_container.empty() && nodes_container.back()) {
        nodes_container.back()->clear_utensor();
      }
      nodes_container.clear();

      // Reset base nodes
      for (auto& node : base_nodes) {
        if (node) {
          node->is_assigned = false;
          node->utensor = UniTensor();
        }
      }
    }

    void build_default_contraction_tree();
    void build_contraction_tree_by_tokens(const std::map<std::string, cytnx_uint64>& name2pos,
                                          const std::vector<std::string>& tokens);
  };
  /// @endcond
}  // namespace cytnx
#endif  // BACKEND_TORCH

#endif  // CYTNX_CONTRACTION_TREE_H_
