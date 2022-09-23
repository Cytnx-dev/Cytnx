#include "ncon.hpp"

using namespace std;

namespace cytnx {
  UniTensor ncon(const std::vector<UniTensor> &tensor_list_in,
                 const std::vector<std::vector<cytnx_int64>> &connect_list_in,
                 const bool check_network /*= false*/, const bool optimize /*= false*/,
                 std::vector<cytnx_int64> cont_order /*= std::vector<cytnx_int64>()*/,
                 const std::vector<cytnx_int64> &out_labels /*= std::vector<cytnx_int64>()*/) {
    string net_in = "";
    map<cytnx_int64, vector<cytnx_uint64>> posbond2tensor;
    for (cytnx_uint64 i = 0; i < tensor_list_in.size(); i++) {
      net_in.append("t"), net_in.append(to_string(i)), net_in.append(": ;");
      vector<cytnx_int64> bds = connect_list_in[i];
      for (cytnx_uint64 j = 0; j < bds.size(); j++) {
        net_in.append(to_string(bds[j]));
        if (j != bds.size() - 1) net_in.append(",");
      }
      net_in.append("\n");
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
                          "[Error][ncon][RegularNetwork] connect list contains not exactly two "
                          "positive same number%s",
                          "\n");
        }
      }
    }
    stack<string> st;
    vector<bool> vis(tensor_list_in.size(), 0);
    for (cytnx_uint64 i = 0; i < cont_order.size(); i++) {
      cytnx_int64 ta = posbond2tensor[cont_order[i]][0], tb = posbond2tensor[cont_order[i]][1];
      st.push("*");
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
    cytnx_int64 oprcnt = 0, needed_parentheses = tensor_list_in.size() - 1;
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
    str_order = "ORDER: " + str_order;
    vector<cytnx_int64> outlbl2;
    for (cytnx_uint64 i = 0; i < connect_list_in.size(); i++) {
      for (cytnx_uint64 j = 0; j < connect_list_in[i].size(); j++) {
        if (connect_list_in[i][j] <= 0) outlbl2.push_back(-connect_list_in[i][j]);
      }
    }
    sort(outlbl2.begin(), outlbl2.end());
    string str_tout = "TOUT: ;";  // Only row space labels
    for (cytnx_uint64 i = 0; i < outlbl2.size(); i++) {
      str_tout.append("-" + to_string(outlbl2[i]));
      if (i != outlbl2.size() - 1) str_tout.append(",");
    }
    str_tout.append("\n");
    UniTensor out;
    Network N;
    // cout << net_in + str_tout + str_order << '\n';
    N.FromString(str_split(net_in + str_tout + str_order, true, "\n"));
    for (cytnx_uint64 i = 0; i < tensor_list_in.size(); i++) {
      N.PutUniTensor("t" + to_string(i), tensor_list_in[i]);
    }
    // cout << N;
    if (!optimize) {
      out = N.Launch(false, str_order.substr(7));  // Remove "ORDER: "
    } else {
      out = N.Launch(true);
    }
    if (!out_labels.empty()) out.set_labels(out_labels);
    return out;
  }
}  // namespace cytnx