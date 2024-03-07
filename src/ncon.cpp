#include "ncon.hpp"

using namespace std;

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  UniTensor ncon(const std::vector<UniTensor> &tensor_list_in,
                 const std::vector<std::vector<cytnx_int64>> &connect_list_in,
                 const bool check_network /*= false*/, const bool optimize /*= false*/,
                 std::vector<cytnx_int64> cont_order /*= std::vector<cytnx_int64>()*/,
                 const std::vector<std::string> &out_labels /*= std::vector<std::string>()*/) {
    vector<string> alias;
    vector<vector<string>> labels;
    map<cytnx_int64, vector<cytnx_uint64>> posbond2tensor;
    for (cytnx_uint64 i = 0; i < tensor_list_in.size(); i++) {
      string name = "t" + to_string(i);
      alias.push_back(name);

      vector<string> label;
      for (cytnx_uint64 j = 0; j < connect_list_in[i].size(); j++) {
        label.push_back(to_string(connect_list_in[i][j]));
      }
      labels.push_back(label);
    }
    vector<cytnx_int64> positive;
    for (cytnx_uint64 i = 0; i < connect_list_in.size(); i++) {
      for (cytnx_uint64 j = 0; j < connect_list_in[i].size(); j++) {
        if (connect_list_in[i][j] > 0) {
          positive.push_back(connect_list_in[i][j]);
          posbond2tensor[connect_list_in[i][j]].push_back(i);
        }
      }
    }
    if (cont_order.empty()) cont_order = vec_unique(positive);  // This is sorted
    for (cytnx_uint64 i = 0; i < connect_list_in.size(); i++) {
      for (cytnx_uint64 j = 0; j < connect_list_in[i].size(); j++) {
        if (connect_list_in[i][j] > 0) {
          cytnx_error_msg(check_network and posbond2tensor[connect_list_in[i][j]].size() != 2,
                          "[Error][ncon] connect list contains not exactly two "
                          "positive same number%s",
                          "\n");
        }
      }
    }
    stack<string> st;
    vector<bool> vis(tensor_list_in.size(), 0);
    for (cytnx_uint64 i = 0; i < cont_order.size(); i++) {
      cytnx_int64 ta = posbond2tensor[cont_order[i]][0], tb = posbond2tensor[cont_order[i]][1];
      if (!vis[ta] or !vis[tb]) st.push("*");
      if (!vis[ta]) st.push("t" + to_string(ta));
      if (!vis[tb]) st.push("t" + to_string(tb));
      vis[ta] = vis[tb] = 1;
    }
    for (cytnx_uint64 i = 0; i < tensor_list_in.size(); i++) {
      if (!vis[i]) {
        st.push("*");
        st.push("t" + to_string(i));
        vis[i] = 1;
      }
    }
    string str_order = "";
    if (!st.empty()) str_order.append(st.top()), st.pop();
    string op[2];
    cytnx_int64 oprcnt = 0;
    cytnx_int64 needed_parentheses = tensor_list_in.size() - 1;
    while (!st.empty()) {
      string ele = st.top();
      st.pop();
      if (ele != "*") {
        op[oprcnt] = ele;
        oprcnt += 1;
      } else {
        if (oprcnt == 2) {
          str_order.append(",(" + op[0] + "," + op[1] + "))");
          needed_parentheses -= 1;
        } else if (oprcnt == 1) {
          str_order.append("," + op[0] + ")");
        }
        oprcnt = 0;
      }
    }
    str_order = string(needed_parentheses, '(').append(str_order);
    // std::cout << str_order << std::endl;
    UniTensor out;
    Network N;
    N.construct(alias, labels, out_labels, 1, str_order, optimize);
    for (cytnx_uint64 i = 0; i < tensor_list_in.size(); i++) {
      N.PutUniTensor(i, tensor_list_in[i]);
    }
    if (!optimize) {
      out = N.Launch();
    } else {
      N.setOrder(true, "");
      out = N.Launch();
    }
    if (!out_labels.empty()) out.set_labels(out_labels);
    return out;
  }
}  // namespace cytnx
#endif
