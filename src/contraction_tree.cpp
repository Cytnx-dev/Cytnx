#include "contraction_tree.hpp"

#include <stack>

#include "utils/order_parser.hpp"

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  void ContractionTree::build_default_contraction_tree() {
    this->reset_contraction_order();

    cytnx_error_msg(this->base_nodes.size() < 2, "[ERROR] Need at least 2 tensors for contraction",
                    "\n");

    std::shared_ptr<Node> left = this->base_nodes[0];
    std::shared_ptr<Node> right;

    this->nodes_container.reserve(this->base_nodes.size());

    for (cytnx_uint64 i = 1; i < this->base_nodes.size(); i++) {
      right = this->base_nodes[i];

      auto new_node = std::make_shared<Node>(left, right);

      this->nodes_container.push_back(new_node);
      left = new_node;
    }

    if (!nodes_container.empty()) {
      auto root = nodes_container.back();
      root->set_root_ptrs();
    }
  }

  void ContractionTree::build_contraction_tree_by_tokens(
    const std::map<std::string, cytnx_uint64> &name2pos, const std::vector<std::string> &tokens) {
    this->reset_contraction_order();
    cytnx_error_msg(this->base_nodes.size() < 2,
                    "[ERROR][ContractionTree][build_contraction_order_by_tokens] contraction tree "
                    "should contain >=2 tensors in order to build contraction order.%s",
                    "\n");
    cytnx_error_msg(
      tokens.size() == 0,
      "[ERROR][ContractionTree][build_contraction_order_by_tokens] Cannot have empty tokens.%s",
      "\n");

    std::string order_line;
    for (const std::string &token : tokens) order_line += token;
    const std::vector<std::string> validated_tokens =
      network_internal::parse_order_line(order_line, 0);

    std::vector<bool> seen_tensors(this->base_nodes.size(), false);
    std::size_t leaf_count = 0;
    for (const std::string &raw_token : validated_tokens) {
      const std::string token = str_strip(raw_token);
      if (token.empty() || token == "(" || token == ")" || token == ",") continue;

      const auto name_position = name2pos.find(token);
      cytnx_error_msg(name_position == name2pos.end(),
                      "[ERROR][ContractionTree] ORDER contains undefined tensor name: %s.\n",
                      token.c_str());
      const std::size_t tensor_index = name_position->second;
      cytnx_error_msg(tensor_index >= seen_tensors.size(),
                      "[ERROR][ContractionTree] ORDER tensor index is out of range: %zu.\n",
                      tensor_index);
      cytnx_error_msg(seen_tensors[tensor_index],
                      "[ERROR][ContractionTree] ORDER contains duplicate tensor name: %s.\n",
                      token.c_str());
      seen_tensors[tensor_index] = true;
      ++leaf_count;
    }
    cytnx_error_msg(
      leaf_count != this->base_nodes.size(),
      "[ERROR][ContractionTree] ORDER must contain every tensor exactly once; found %zu tensor "
      "names for %zu tensors.\n",
      leaf_count, this->base_nodes.size());

    std::stack<std::shared_ptr<Node>> stk;
    std::shared_ptr<Node> left, right;
    std::stack<char> operators;
    char topc;
    std::size_t pos = 0;
    std::string tok;

    // evaluate each token, and construct the Contraction Tree.
    this->nodes_container.reserve(
      this->base_nodes.size());  // reserve a contiguous memeory address to prevent re-allocate that
                                 // change address.
    for (cytnx_uint64 i = 0; i < validated_tokens.size(); i++) {
      tok = str_strip(validated_tokens[i]);  // remove space.
      if (tok.length() == 0) continue;
      if (tok == "(") {
        operators.push(tok.c_str()[0]);
      } else if (tok == ")") {
        if (!operators.empty()) {
          topc = operators.top();
          while ((topc != '(')) {
            operators.pop();
            right = stk.top();
            stk.pop();
            left = stk.top();
            stk.pop();
            auto new_node = std::make_shared<Node>(left, right);
            this->nodes_container.push_back(new_node);
            stk.push(this->nodes_container.back());
            if (!operators.empty())
              topc = operators.top();
            else
              break;
          }
        }
        operators.pop();  // discard the '('
      } else if (tok == ",") {
        if (!operators.empty()) {
          topc = operators.top();
          while ((topc != '(') && (topc != ')')) {
            operators.pop();
            right = stk.top();
            stk.pop();
            left = stk.top();
            stk.pop();
            auto new_node = std::make_shared<Node>(left, right);
            this->nodes_container.push_back(new_node);
            stk.push(this->nodes_container.back());
            if (!operators.empty())
              topc = operators.top();
            else
              break;
          }
        }
        operators.push(',');
      } else {
        cytnx_uint64 idx;
        try {
          idx = name2pos.at(tok);
        } catch (const std::out_of_range &) {
          cytnx_error_msg(true,
                          "[ERROR][ContractionTree][build_contraction_order_by_token] tokens "
                          "contain invalid TN name: %s ,which is not previously defined. \n",
                          tok.c_str());
        }
        stk.push(this->base_nodes[idx]);
      }

    }  // for each token

    while (!operators.empty()) {
      operators.pop();
      right = stk.top();
      stk.pop();
      left = stk.top();
      stk.pop();
      // this->nodes_container.back().name = right->name +  left->name;
      auto new_node = std::make_shared<Node>(left, right);
      this->nodes_container.push_back(new_node);
      stk.push(this->nodes_container.back());
    }

    cytnx_error_msg(stk.size() != 1 || this->nodes_container.size() + 1 != leaf_count,
                    "[ERROR][ContractionTree] ORDER did not produce one complete binary tree.%s",
                    "\n");
  }

}  // namespace cytnx
#endif
