#include <typeinfo>
#include "Network.hpp"
#include "utils/utils_internal_interface.hpp"
#include <stack>
#include <algorithm>
#include <iostream>

using namespace std;

namespace cytnx {

  // these two are internal functions:
  void _parse_ORDER_line_(vector<string> &tokens, const string &line) {
    cytnx_error_msg((line.find_first_of("\t;\n:") != string::npos),
                    "[ERROR][Network][Fromfile] invalid ORDER line format.%s", "\n");
    cytnx_error_msg((line.find_first_of("(),") == string::npos),
                    "[ERROR][Network][Fromfile] invalid ORDER line format.%s",
                    " tensors should be seperate by delimiter \',\' (comma), and/or wrapped with "
                    "\'(\' and \')\'");

    // check mismatch:
    size_t lbrac_n = std::count(line.begin(), line.end(), '(');
    size_t rbrac_n = std::count(line.begin(), line.end(), ')');
    cytnx_error_msg(lbrac_n != rbrac_n, "[ERROR][Network][Fromfile] parentheses mismatch.%s", "\n");

    // slice the line into pieces by parentheses and comma
    tokens = str_findall(line, "(),");

    cytnx_error_msg(tokens.size() == 0, "[ERROR][Network][Fromfile] invalid ORDER line.%s", "\n");
  }
  void _parse_TOUT_line_(vector<cytnx_int64> &lbls, cytnx_uint64 &TOUT_iBondNum,
                         const string &line) {
    lbls.clear();
    vector<string> tmp = str_split(line, false, ";");
    cytnx_error_msg(tmp.size() != 2, "[ERROR][Network] Fromfile: %s\n", "Invalid TOUT line");

    // handle col-space lbl
    vector<string> ket_lbls = str_split(tmp[0], false, ",");
    if (ket_lbls.size() == 1)
      if (ket_lbls[0].length() == 0) ket_lbls.clear();
    for (cytnx_uint64 i = 0; i < ket_lbls.size(); i++) {
      string tmp = str_strip(ket_lbls[i]);
      cytnx_error_msg(tmp.length() == 0,
                      "[ERROR][Network][Fromfile] Invalid labels for TOUT line.%s", "\n");
      cytnx_error_msg((tmp.find_first_not_of("0123456789-") != string::npos),
                      "[ERROR][Network] Fromfile: %s\n",
                      "Invalid TOUT line. label contain non integer.");
      lbls.push_back(stoi(tmp, nullptr));
    }
    TOUT_iBondNum = lbls.size();

    // handle row-space lbl
    vector<string> bra_lbls = str_split(tmp[1], false, ",");
    if (bra_lbls.size() == 1)
      if (bra_lbls[0].length() == 0) bra_lbls.clear();
    for (cytnx_uint64 i = 0; i < bra_lbls.size(); i++) {
      string tmp = str_strip(bra_lbls[i]);
      cytnx_error_msg(tmp.length() == 0,
                      "[ERROR][Network][Fromfile] Invalid labels for TOUT line.%s", "\n");
      cytnx_error_msg((tmp.find_first_not_of("0123456789-") != string::npos),
                      "[ERROR][Network] Fromfile: %s\n",
                      "Invalid TOUT line. label contain non integer.");
      lbls.push_back(stoi(tmp, nullptr));
    }
  }
  void _parse_TN_line_(vector<cytnx_int64> &lbls, cytnx_uint64 &TN_iBondNum, const string &line) {
    lbls.clear();
    vector<string> tmp = str_split(line, false, ";");
    cytnx_error_msg(tmp.size() != 2, "[ERROR][Network] Fromfile: %s\n", "Invalid TN line");

    // handle col-space lbl
    vector<string> ket_lbls = str_split(tmp[0], false, ",");
    if (ket_lbls.size() == 1)
      if (ket_lbls[0].length() == 0) ket_lbls.clear();
    for (cytnx_uint64 i = 0; i < ket_lbls.size(); i++) {
      string tmp = str_strip(ket_lbls[i]);
      cytnx_error_msg(tmp.length() == 0, "[ERROR][Network][Fromfile] Invalid labels for TN line.%s",
                      "\n");
      cytnx_error_msg((tmp.find_first_not_of("0123456789-") != string::npos),
                      "[ERROR][Network] Fromfile: %s\n",
                      "Invalid TN line. label contain non integer.");
      lbls.push_back(stoi(tmp, nullptr));
    }
    TN_iBondNum = lbls.size();

    // handle row-space lbl
    vector<string> bra_lbls = str_split(tmp[1], false, ",");
    if (bra_lbls.size() == 1)
      if (bra_lbls[0].length() == 0) bra_lbls.clear();
    for (cytnx_uint64 i = 0; i < bra_lbls.size(); i++) {
      string tmp = str_strip(bra_lbls[i]);
      cytnx_error_msg(tmp.length() == 0,
                      "[ERROR][Network][Fromfile] Invalid labels for TOUT line.%s", "\n");
      cytnx_error_msg((tmp.find_first_not_of("0123456789-") != string::npos),
                      "[ERROR][Network] Fromfile: %s\n",
                      "Invalid TN line. label contain non integer.");
      lbls.push_back(stoi(tmp, nullptr));
    }

    cytnx_error_msg(lbls.size() == 0, "[ERROR][Network][Fromfile] %s\n",
                    "Invalid TN line. no label present in this line, which is invalid.%s", "\n");
  }

  void RegularNetwork::Fromfile(const std::string &fname) {
    const cytnx_uint64 MAXLINES = 1024;

    // empty all
    this->Clear();

    // open file
    std::ifstream infile;
    infile.open(fname.c_str());
    if (!(infile.is_open())) {
      cytnx_error_msg(true, "[Network] Error in opening file \'", fname.c_str(), "\'.\n");
    }

    string line;
    cytnx_uint64 lnum = 0;
    vector<string> tmpvs;

    // read each line:
    while (lnum < MAXLINES) {
      lnum++;
      getline(infile, line);
      line = str_strip(line, "\n");  // remove \n on each end.
      line = str_strip(line);  // remove space on each end
      if (infile.eof()) break;

      // check :
      // cout << "line:" << lnum << "|" << line <<"|"<< endl;
      if (line.length() == 0) continue;  // blank line
      if (line.at(0) == '#') continue;  // comment whole line.

      // remove any comment at eol :
      line = str_split(line, true, "#")[0];  // remove comment on end.

      // A valid line should contain ':':
      cytnx_error_msg(
        line.find_first_of(":") == string::npos,
        "[ERROR][Network][Fromfile] invalid line in network file at line: %d. should contain \':\'",
        lnum);

      tmpvs = str_split(line, false, ":");  // note that checking empty string!
      cytnx_error_msg(tmpvs.size() != 2,
                      "[ERROR][Network] invalid line in network file at line: %d. \n", lnum);

      string name = str_strip(tmpvs[0]);
      string content = str_strip(tmpvs[1]);

      // check if name contain invalid keyword or not assigned.
      cytnx_error_msg(name.length() == 0,
                      "[ERROR][Network][Fromfile] invalid tensor name at line: %d\n", lnum);
      cytnx_error_msg(name.find_first_of(" ;,") != string::npos,
                      "[ERROR] invalid Tensor name at line %d\n", lnum);

      // dispatch:
      if (name == "ORDER") {
        if (content.length()) {
          // cut the line into tokens,
          // and leave it to process by CtTree after read all lines.
          _parse_ORDER_line_(this->ORDER_tokens, content);
        }
      } else if (name == "TOUT") {
        // if content has length, then pass to process.
        if (content.length()) {
          // this is an internal function that is defined in this cpp file.
          _parse_TOUT_line_(this->TOUT_labels, this->TOUT_iBondNum, content);
        }
      } else {
        this->names.push_back(name);
        // check if name is valid:
        if (name2pos.find(name) != name2pos.end()) {
          cytnx_error_msg(true,
                          "[ERROR][Network][Fromfile] tensor name: [%s] has already exist. Cannot "
                          "have duplicated tensor name in a network.",
                          name.c_str());
        }
        cytnx_error_msg(name.find_first_of("\t;\n: ") != string::npos,
                        "[ERROR][Network][Fromfile] invalid tensor name. cannot contain [' '] or "
                        "[';'] or [':'] or ['\\t'] .%s",
                        "\n");

        // check exists content:
        cytnx_error_msg(content.length() == 0,
                        "[ERROR][Network][Fromfile] invalid tensor labelsat line %d. cannot have "
                        "empty labels for input tensor. \n",
                        lnum);

        this->name2pos[name] = names.size() - 1;  // register
        this->label_arr.push_back(vector<cytnx_int64>());
        cytnx_uint64 tmp_iBN;
        // this is an internal function that is defined in this cpp file.
        _parse_TN_line_(this->label_arr.back(), tmp_iBN, content);
        this->iBondNums.push_back(tmp_iBN);
      }

    }  // end readlines
    infile.close();

    cytnx_error_msg(
      lnum >= MAXLINES,
      "[ERROR][Network][Fromfile] network file exceed the maxinum allowed lines, MAXLINES=2048%s",
      "\n");

    cytnx_error_msg(
      this->names.size() < 2,
      "[ERROR][Network][Fromfile] invalid network file. Should have at least 2 tensors defined.%s",
      "\n");

    this->tensors.resize(this->names.size());
    this->CtTree.base_nodes.resize(this->names.size());

    // contraction order:
    if (ORDER_tokens.size() != 0) {
      CtTree.build_contraction_order_by_tokens(this->name2pos, ORDER_tokens);
    } else {
      CtTree.build_default_contraction_order();
    }
  }

  void RegularNetwork::PutUniTensor(const cytnx_uint64 &idx, const UniTensor &utensor,
                                    const bool &is_clone) {
    cytnx_error_msg(idx >= this->CtTree.base_nodes.size(),
                    "[ERROR][RegularNetwork][PutUniTensor] index=%d out of range.\n", idx);

    // check shape:
    cytnx_error_msg(this->label_arr[idx].size() != utensor.rank(),
                    "[ERROR][RegularNetwork][PutUniTensor] tensor name: [%s], the rank of input "
                    "UniTensor does not match the definition in network file.\n",
                    this->names[idx].c_str());
    cytnx_error_msg(this->iBondNums[idx] != utensor.Rowrank(),
                    "[ERROR][RegularNetwork][PutUniTensor] tensor name: [%s], the row-rank of "
                    "input UniTensor does not match the semicolon defined in network file.\n",
                    this->names[idx].c_str());

    if (is_clone) {
      this->tensors[idx] = utensor.clone();
    } else {
      this->tensors[idx] = utensor;
    }
  }

  void RegularNetwork::PutUniTensor(const std::string &name, const UniTensor &utensor,
                                    const bool &is_clone) {
    cytnx_uint64 idx;
    try {
      idx = this->name2pos.at(name);
    } catch (std::out_of_range) {
      cytnx_error_msg(true,
                      "[ERROR][RegularNetwork][PutUniTensor] cannot find the tensor name: [%s] in "
                      "current network.\n",
                      name.c_str());
    }

    this->PutUniTensor(idx, utensor, is_clone);
  }

  UniTensor RegularNetwork::Launch() {
    // 1. check tensors are all set, and put all unitensor on node for contraction:
    cytnx_error_msg(this->tensors.size() == 0,
                    "[ERROR][Launch][RegularNetwork] cannot launch an un-initialize network.%s",
                    "\n");

    vector<vector<cytnx_int64>> old_labels;
    for (cytnx_uint64 idx = 0; idx < this->tensors.size(); idx++) {
      cytnx_error_msg(this->tensors[idx].uten_type() == UTenType.Void,
                      "[ERROR][Launch][RegularNetwork] tensor at [%d], name: [%s] is not set.\n",
                      idx, this->names[idx].c_str());
      // transion save old labels:
      old_labels.push_back(this->tensors[idx].labels());

      // modify the label of unitensor (shared):
      this->tensors[idx].set_labels(this->label_arr[idx]);

      this->CtTree.base_nodes[idx].utensor = this->tensors[idx];
      this->CtTree.base_nodes[idx].is_assigned = true;
    }

    // 2. contract using postorder traversal:
    // cout << this->CtTree.nodes_container.size() << endl;
    stack<Node *> stk;
    Node *root = &(this->CtTree.nodes_container.back());
    int ly = 0;
    bool ict;

    do {
      // move the lmost
      while ((root != nullptr)) {
        if (root->right != nullptr) stk.push(root->right);
        stk.push(root);
        root = root->left;
      }

      root = stk.top();
      stk.pop();
      // cytnx_error_msg(stk.size()==0,"[eRROR]","\n");
      ict = true;
      if ((root->right != nullptr) && !stk.empty()) {
        if (stk.top() == root->right) {
          stk.pop();
          stk.push(root);
          root = root->right;
          ict = false;
        }
      }
      if (ict) {
        // process!

        // cout << "OK" << endl;
        if ((root->right != nullptr) && (root->left != nullptr)) {
          root->utensor = Contract(root->left->utensor, root->right->utensor);
          root->left->clear_utensor();  // remove intermediate unitensor to save heap space
          root->right->clear_utensor();  // remove intermediate unitensor to save heap space
          root->is_assigned = true;
          // cout << "contract!" << endl;
        }

        root = nullptr;
      }

      // cout.flush();
      // break;

    } while (!stk.empty());

    // 3. get result:
    UniTensor out = this->CtTree.nodes_container.back().utensor;

    // 4. reset nodes:
    this->CtTree.reset_nodes();

    // 5. reset back the original labels:
    for (cytnx_uint64 i = 0; i < this->tensors.size(); i++) {
      this->tensors[i].set_labels(old_labels[i]);
    }

    // 6. permute accroding to pre-set labels:
    if (TOUT_labels.size()) {
      out.permute_(TOUT_labels, TOUT_iBondNum, true);
    }

    // UniTensor out;
    return out;
  }

}  // namespace cytnx
