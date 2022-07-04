#include "contraction_tree.hpp"

#include <stack>

using namespace std;

namespace cytnx {
  void ContractionTree::build_default_contraction_tree() {
    this->reset_contraction_order();
    cytnx_error_msg(this->base_nodes.size() < 2,
                    "[ERROR][ContractionTree][build_default_contraction_order] contraction tree "
                    "should contain >=2 tensors in order to build contraction order.%s",
                    "\n");

    Node *left = &(this->base_nodes[0]);
    Node *right;
    this->nodes_container.reserve(
      this->base_nodes.size());  // reserve a contiguous memeory address to prevent re-allocate that
                                 // change address.
    for (cytnx_uint64 i = 1; i < this->base_nodes.size(); i++) {
      right = &(this->base_nodes[i]);
      this->nodes_container.push_back(Node(left, right));
      left = &(this->nodes_container.back());
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
      "[ERROR][ContractionTree][build_contraction_order_by_tokens] cannot have empty tokens.%s",
      "\n");

    stack<Node *> stk;
    Node *left;
    Node *right;
    stack<char> operators;
    char topc;
    size_t pos = 0;
    std::string tok;

    // evaluate each token, and construct the Contraction Tree.
    this->nodes_container.reserve(
      this->base_nodes.size());  // reserve a contiguous memeory address to prevent re-allocate that
                                 // change address.
    for (cytnx_uint64 i = 0; i < tokens.size(); i++) {
      tok = str_strip(tokens[i]);  // remove space.
      // cout << tokens[i] << "|" << tok << "|" << endl;
      if (tok.length() == 0) continue;
      // cout << tok << "|";
      if (tok == "(") {
        operators.push(tok.c_str()[0]);
        // cout << "put(" << endl;
      } else if (tok == ")") {
        // cout << "put)-->";
        if (!operators.empty()) {
          topc = operators.top();
          while ((topc != '(')) {
            operators.pop();
            right = stk.top();
            stk.pop();
            left = stk.top();
            stk.pop();
            this->nodes_container.push_back(Node(left, right));
            // cout << right->name << " " << left->name <<">";
            // this->nodes_container.back().name = right->name + left->name;
            stk.push(&this->nodes_container.back());
            if (!operators.empty())
              topc = operators.top();
            else
              break;
          }
        }
        // cout << endl;
        operators.pop();  // discard the '('
      } else if (tok == ",") {
        // cout << "put,-->";
        if (!operators.empty()) {
          topc = operators.top();
          while ((topc != '(') && (topc != ')')) {
            operators.pop();
            right = stk.top();
            stk.pop();
            left = stk.top();
            stk.pop();
            this->nodes_container.push_back(Node(left, right));
            // cout << right->name << " " << left->name << ">";
            // this->nodes_container.back().name = right->name  + left->name;
            stk.push(&this->nodes_container.back());
            if (!operators.empty())
              topc = operators.top();
            else
              break;
          }
        }
        // cout << endl;
        operators.push(',');
      } else {
        cytnx_uint64 idx;
        try {
          idx = name2pos.at(tok);
        } catch (std::out_of_range) {
          cytnx_error_msg(true,
                          "[ERROR][ContractionTree][build_contraction_order_by_token] tokens "
                          "contain invalid TN name: %s ,which is not previously defined. \n",
                          tok.c_str());
        }
        stk.push(&this->base_nodes[idx]);
        // cout << "TN" << this->base_nodes[idx].name << endl;
      }

    }  // for each token

    while (!operators.empty()) {
      operators.pop();
      right = stk.top();
      stk.pop();
      left = stk.top();
      stk.pop();
      // this->nodes_container.back().name = right->name +  left->name;
      this->nodes_container.push_back(Node(left, right));
      stk.push(&this->nodes_container.back());
    }
    /*
    cout << "============" << endl;
    for(int i=0;i<this->nodes_container.size();i++){
        cout << this->nodes_container[i].name << endl;
    }
    cout << "============" << endl;
    */
  }

}  // namespace cytnx
