#ifndef _H_contraction_tree_
#define _H_contraction_tree_

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
  class Node : public std::enable_shared_from_this<Node> {
   public:
    UniTensor utensor;
    bool is_assigned;
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;
    std::string name;
    std::shared_ptr<Node> root;

    Node() : is_assigned(false), left(nullptr), right(nullptr), root(nullptr) {}
    Node(const Node &rhs)
        : utensor(rhs.utensor),
          is_assigned(rhs.is_assigned),
          left(rhs.left),
          right(rhs.right),
          root(rhs.root) {}
    Node &operator=(const Node &rhs) {
      if (this != &rhs) {
        utensor = rhs.utensor;
        is_assigned = rhs.is_assigned;
        left = rhs.left;
        right = rhs.right;
        root = rhs.root;
      }
      return *this;
    }
    Node(std::shared_ptr<Node> in_left, std::shared_ptr<Node> in_right,
         const UniTensor &in_uten = UniTensor())
        : is_assigned(false), left(in_left), right(in_right), root(nullptr) {
      if (in_left) in_left->root = shared_from_this();
      if (in_right) in_right->root = shared_from_this();
      if (in_uten.uten_type() != UTenType.Void) utensor = in_uten;
    }
    void assign_utensor(const UniTensor &in_uten) {
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
    std::vector<Node> nodes_container;
    std::vector<Node> base_nodes;

    ContractionTree() = default;
    ContractionTree(const ContractionTree &rhs)
        : nodes_container(rhs.nodes_container), base_nodes(rhs.base_nodes) {}
    ContractionTree &operator=(const ContractionTree &rhs) {
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
    void reset_contraction_order() {
      nodes_container.clear();
      for (auto &node : base_nodes) {
        node.root = nullptr;
      }
    }
    void reset_nodes() {
      for (auto &node : nodes_container) {
        node.clear_utensor();
      }
      for (auto &node : base_nodes) {
        node.clear_utensor();
      }
    }
    void build_default_contraction_tree();
    void build_contraction_tree_by_tokens(const std::map<std::string, cytnx_uint64> &name2pos,
                                          const std::vector<std::string> &tokens);
  };
}  // namespace cytnx
#endif  // BACKEND_TORCH

#endif
