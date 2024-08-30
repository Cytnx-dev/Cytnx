#include <typeinfo>
#include "Network.hpp"
#include "search_tree.hpp"
#include <stack>
#include <algorithm>
#include <iostream>
#include "Generator.hpp"
using namespace std;

#ifdef BACKEND_TORCH
#else
  #include "utils/cutensornet.hpp"

namespace cytnx {

  // these two are internal functions:
  void _parse_ORDER_line_(vector<string> &tokens, const string &line,
                          const cytnx_uint64 &line_num) {
    cytnx_error_msg((line.find_first_of("\t;\n:") != string::npos),
                    "[ERROR][Network][Fromfile] line:%d invalid ORDER line format.%s", line_num,
                    "\n");
    cytnx_error_msg((line.find_first_of("(),") == string::npos),
                    "[ERROR][Network][Fromfile] line:%d invalid ORDER line format.%s", line_num,
                    " tensors should be seperate by delimiter \',\' (comma), and/or wrapped with "
                    "\'(\' and \')\'");

    // check mismatch:
    size_t lbrac_n = count(line.begin(), line.end(), '(');
    size_t rbrac_n = count(line.begin(), line.end(), ')');
    cytnx_error_msg(lbrac_n != rbrac_n, "[ERROR][Network][Fromfile] parentheses mismatch.%s", "\n");

    // slice the line into pieces by parentheses and comma
    tokens = str_findall(line, "(),");

    cytnx_error_msg(tokens.size() == 0, "[ERROR][Network][Fromfile] line:%d invalid ORDER line.%s",
                    line_num, "\n");
  }

  void _parse_TOUT_line_(vector<string> &labels, cytnx_uint64 &TOUT_iBondNum, const string &line,
                         const cytnx_uint64 &line_num) {
    labels.clear();

    vector<string> tmp = str_split(line, false, ";");
    // cytnx_error_msg(tmp.size() != 2, "[ERROR][Network][Fromfile] line:%d %s\n", line_num,
    //                 "Invalid TOUT line");
    if (tmp.size() == 1) {
      // no ; provided
      vector<string> tout_lbls = str_split(line, false, ",");
      cytnx_uint64 default_rowrank = tout_lbls.size() / 2;
      // handle col-space label
      for (cytnx_uint64 i = 0; i < default_rowrank; i++) {
        string tmp_ = str_strip(tout_lbls[i]);
        cytnx_error_msg(tmp_.length() == 0,
                        "[ERROR][Network][Fromfile] line:%d Invalid labels for TOUT line.%s",
                        line_num, "\n");
        labels.push_back(tmp_);
      }
      TOUT_iBondNum = labels.size();
      // handle row-space label
      for (cytnx_uint64 i = default_rowrank; i < tout_lbls.size(); i++) {
        string tmp_ = str_strip(tout_lbls[i]);
        cytnx_error_msg(tmp_.length() == 0,
                        "[ERROR][Network][Fromfile] line:%d Invalid labels for TOUT line.%s",
                        line_num, "\n");
        labels.push_back(tmp_);
      }
    } else if (tmp.size() == 2) {
      // one ;
      // handle col-space label
      vector<string> ket_labels = str_split(tmp[0], false, ",");
      if (ket_labels.size() == 1)
        if (ket_labels[0].length() == 0) ket_labels.clear();
      for (cytnx_uint64 i = 0; i < ket_labels.size(); i++) {
        string tmp = str_strip(ket_labels[i]);
        cytnx_error_msg(tmp.length() == 0,
                        "[ERROR][Network][Fromfile] line:%d Invalid labels for TOUT line.%s",
                        line_num, "\n");
        labels.push_back(tmp);
      }
      TOUT_iBondNum = labels.size();
      // handle row-space label
      vector<string> bra_labels = str_split(tmp[1], false, ",");
      if (bra_labels.size() == 1)
        if (bra_labels[0].length() == 0) bra_labels.clear();
      for (cytnx_uint64 i = 0; i < bra_labels.size(); i++) {
        string tmp = str_strip(bra_labels[i]);
        cytnx_error_msg(tmp.length() == 0,
                        "[ERROR][Network][Fromfile] line:%d Invalid labels for TOUT line.%s",
                        line_num, "\n");
        labels.push_back(tmp);
      }
    } else if (tmp.size() > 2) {
      // more than one ;
      cytnx_error_msg(true, "[ERROR][Network] line:%d Invalid TOUT line.%s", line_num, "\n");
    }
  }

  /// This is debug function for printing special characters
  void tri(const char *text) {
    for (const char *p = text; *p != '\0'; ++p) {
      int c = (unsigned char)*p;

      switch (c) {
        case '\\':
          printf("\\\\");
          break;
        case '\n':
          printf("\\n");
          break;
        case '\r':
          printf("\\r");
          break;
        case '\t':
          printf("\\t");
          break;

          // TODO: Add other C character escapes here.  See:
          // <https://en.wikipedia.org/wiki/Escape_sequences_in_C#Table_of_escape_sequences>

        default:
          if (isprint(c)) {
            putchar(c);
          } else {
            printf("\\x%X", c);
          }
          break;
      }
    }
  }

  void _parse_TN_line_(vector<string> &labels, cytnx_uint64 &TN_iBondNum, const string &line,
                       const cytnx_uint64 &line_num) {
    labels.clear();
    // vector<string> tmp = str_split(line, false, ";");
    // cytnx_error_msg(tmp.size() != 2, "[ERROR][Network][Fromfile] line:%d %s\n", line_num,
    //                 "Invalid TN line. A \';\' should be used to indicate the rowrank.\nexample1>
    //                 "
    //                 "\'Tn: 0, 1; 2, 3\'\nexample2> \'Tn: ; -1, 2, 3\'");

    // // handle col-space label
    // vector<string> ket_labels = str_split(tmp[0], false, ",");
    // if (ket_labels.size() == 1)
    //   if (ket_labels[0].length() == 0) ket_labels.clear();
    // for (cytnx_uint64 i = 0; i < ket_labels.size(); i++) {
    //   string tmp = str_strip(ket_labels[i]);
    //   cytnx_error_msg(tmp.length() == 0,
    //                   "[ERROR][Network][Fromfile] line:%d Invalid labels for TN line.%s",
    //                   line_num,
    //                   "\n");
    //   cytnx_error_msg((tmp.find_first_not_of("0123456789-") != string::npos),
    //                   "[ERROR][Network][Fromfile] line:%d %s\n", line_num,
    //                   "Invalid TN line. label contain non integer.");
    //   labels.push_back(tmp);
    // }
    // TN_iBondNum = labels.size();

    // // handle row-space label
    // vector<string> bra_labels = str_split(tmp[1], false, ",");
    // if (bra_labels.size() == 1)
    //   if (bra_labels[0].length() == 0) bra_labels.clear();
    // for (cytnx_uint64 i = 0; i < bra_labels.size(); i++) {
    //   string tmp = str_strip(bra_labels[i]);
    //   cytnx_error_msg(tmp.length() == 0,
    //                   "[ERROR][Network][Fromfile] line:%d Invalid labels for TOUT line.%s",
    //                   line_num, "\n");

    //   // tri(tmp.c_str());

    //   // cout << tmp.size() << endl;
    //   // cout << tmp.find_first_not_of("0123456789-") << endl;
    //   cytnx_error_msg((tmp.find_first_not_of("0123456789-") != string::npos),
    //                   "[ERROR][Network][Fromfile] line:%d %s\n", line_num,
    //                   "Invalid TN line. label contain non integer.");
    //   labels.push_back(tmp);
    // }

    vector<string> alllabels = str_split(line, false, ",");
    if (alllabels.size() == 1)
      if (alllabels[0].length() == 0) alllabels.clear();
    for (cytnx_uint64 i = 0; i < alllabels.size(); i++) {
      string tmp = str_strip(alllabels[i]);
      cytnx_error_msg(tmp.length() == 0,
                      "[ERROR][Network][Fromfile] line:%d Invalid labels for TN line.%s", line_num,
                      "\n");
      // cytnx_error_msg((tmp.find_first_not_of("0123456789-") != string::npos),
      //                 "[ERROR][Network][Fromfile] line:%d %s\n", line_num,
      //                 "Invalid TN line. label contain non integer.");
      labels.push_back(tmp);
    }

    TN_iBondNum = labels.size();

    cytnx_error_msg(labels.size() == 0, "[ERROR][Network][Fromfile] line:%d %s\n", line_num,
                    "Invalid TN line. no label present in this line, which is invalid.%s", "\n");
  }

  void _extract_TNs_from_ORDER_(vector<string> &TN_names, const vector<string> &tokens) {
    TN_names.clear();
    for (cytnx_uint64 i = 0; i < tokens.size(); i++) {
      string tok = str_strip(tokens[i]);  // remove space.
      if (tok.length() == 0) continue;
      if ((tok != "(") && (tok != ")") && (tok != ",")) {
        TN_names.push_back(tok);
      }
    }
  }

  string einsumpath_to_string(vector<pair<cytnx_int64, cytnx_int64>> path, vector<string> tns) {
    string res;
    for (int i = 0; i < path.size(); i++) {
      int id1 = path[i].first;
      int id2 = path[i].second;
      res.clear();
      res.append("(");
      res.append(tns[id1]);
      res.append(",");
      res.append(tns[id2]);
      res.append(")");
      tns.erase(tns.begin() + id1);
      if (id1 > id2)
        tns.erase(tns.begin() + id2);
      else
        tns.erase(tns.begin() + id2 - 1);
      tns.push_back(res);
    }
    return res;
  }

  /*
   * This function is not used in the codebase and not implemented properly.
   * Currently disabled.
   */
  vector<pair<cytnx_int64, cytnx_int64>> CtTree_to_eisumpath(ContractionTree CtTree,
                                                             vector<string> tns) {
    cytnx_error_msg(true,
                    "[Error][RegularNetwork] CtTree_to_eisumpath is not properly implemented, "
                    "disabled for now.%s",
                    "\n");
    vector<pair<cytnx_int64, cytnx_int64>> path;
    stack<Node *> stk;
    Node *root = &(CtTree.nodes_container.back());
    int ly = 0;
    bool ict;
    do {
      while ((root != nullptr)) {
        if (root->right != nullptr) stk.push(root->right);
        stk.push(root);
        root = root->left;
      }
      root = stk.top();
      stk.pop();
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
        if ((root->right != nullptr) && (root->left != nullptr)) {
          auto it = find(tns.begin(), tns.end(), root->left->name);
          int id1 = it - tns.begin();
          it = find(tns.begin(), tns.end(), root->right->name);
          int id2 = it - tns.begin();
          tns.erase(tns.begin() + id1);
          if (id1 > id2)
            tns.erase(tns.begin() + id2);
          else
            tns.erase(tns.begin() + id2 - 1);
          tns.push_back(root->name);
          path.push_back(std::pair<cytnx_int64, cytnx_int64>(id1, id2));
        }
        root = nullptr;
      }
    } while (!stk.empty());
    // for (int i = 0; i < path.size(); i++) {
    //   std::cout << path[i].first << ", " << path[i].second << std::endl;
    // }
    return path;
  }

  void check(vector<UniTensor> &tns, vector<string> &tn_names) {
    // check tensors are all set, and put all unitensor on node for contraction:
    cytnx_error_msg(
      tns.size() == 0,
      "[ERROR][RegularNetwork] Cannot find optimal order/Launch for an un-initialize network.%s",
      "\n");
    cytnx_error_msg(tns.size() < 2,
                    "[ERROR][RegularNetwork] Network should contain >=2 tensors to find "
                    "optimal order/Launch.%s",
                    "\n");
    for (cytnx_uint64 idx = 0; idx < tns.size(); idx++) {
      cytnx_error_msg(tns[idx].uten_type() == UTenType.Void,
                      "[ERROR][RegularNetwork] tensor at [%d], name: [%s] is not set.\n", idx,
                      tn_names[idx].c_str());
    }

    // check same device and uten_type
    int tn_device = tns[0].device();
    int utentype = tns[0].uten_type();
    for (int i = 1; i < tns.size(); i++) {
      cytnx_error_msg(tns[i].device() != tn_device,
                      "[ERROR][Launch][RegularNetwork] cannot find optimal order/launch with "
                      "tensors on different devices, tensor "
                      "at [0] is on device %d while tensor at [%d] in on device %d. %s",
                      tn_device, i, tns[i].device(), "\n");
      cytnx_error_msg(tns[i].uten_type() != utentype,
                      "[ERROR][Launch][RegularNetwork] cannot find optimal order/launch with "
                      "tensors of different unitensor types, tensor "
                      "at [0] is uten_type %d while tensor at [%d] in uten_type %d. %s",
                      utentype, i, tns[i].uten_type(), "\n");
    }
  }

  void RegularNetwork::Contract_plan(const vector<UniTensor> &utensors, const string &Tout,
                                     const vector<string> &alias, const string &contract_order) {
    cytnx_error_msg(utensors.size() < 2,
                    "[ERROR][Network] invalid network. Should have at least 2 tensors defined.%s",
                    "\n");

    if (contract_order.length()) {
      // checing if alias is set!
      cytnx_error_msg(alias.size() == 0,
                      "[ERRPR] conraction_order need to be specify using alias name, so alias name "
                      "have to be assigned!%s",
                      "\n");
    }

    if (alias.size())
      cytnx_error_msg(utensors.size() != alias.size(),
                      "[ERROR] alias of UniTensor need to be assigned for all utensors.%s", "\n");

    bool isORDER_exist = false;
    // reading
    if (contract_order.length()) {
      // ORDER assigned
      _parse_ORDER_line_(this->ORDER_tokens, contract_order, 0);
      isORDER_exist = true;
    }
    if (Tout.length()) {
      // TOUT assigned
      _parse_TOUT_line_(this->TOUT_labels, this->TOUT_iBondNum, Tout, 0);
    }

    // assign input tensors into slots:
    string name;
    for (unsigned int i = 0; i < utensors.size(); i++) {
      if (alias.size()) {
        this->names.push_back(alias[i]);
        name = alias[i];
      } else {
        if (utensors[i].name().length()) {
          name = utensors[i].name() + "_usr";
        } else {
          name = "uname_T" + to_string(i);
        }
        this->names.push_back(name);
      }

      // check if name is valid:
      if (name2pos.find(name) != name2pos.end()) {
        cytnx_error_msg(true,
                        "[ERROR][Network][Fromfile] tensor name: [%s] has already exist. Cannot "
                        "have duplicated tensor name in a network.",
                        name.c_str());
      }

      this->name2pos[name] = names.size() - 1;  // register
      // cout << name << "|" << names.size() - 1 << endl;
      this->label_arr.push_back(vector<string>());
      cytnx_uint64 tmp_iBN;
      // this is an internal function that is defined in this cpp file.
      this->label_arr.back() = utensors[i].labels();
      this->iBondNums.push_back(utensors[i].rowrank());
      //_parse_TN_line_(this->label_arr.back(),tmp_iBN,content,lnum);
      // this->iBondNums.push_back(tmp_iBN);

    }  // traversal input tensor list

    this->tensors.resize(this->names.size());
    this->CtTree.base_nodes.resize(this->names.size());

    // checking if all TN are set in ORDER.
    //  only alias assigned will activate order
    if (isORDER_exist) {
      vector<string> TN_names;  // this should be integer!
      _extract_TNs_from_ORDER_(TN_names, this->ORDER_tokens);
      cytnx_error_msg(TN_names.size() != utensors.size(),
                      "[ERROR][Network][Contract--planning] order assigned but the [%d] tensors "
                      "appears in ORDER does not match the # input tensors [%d]\n",
                      TN_names.size(), utensors.size());
      for (int i = 0; i < this->names.size(); i++) {
        auto it = find(TN_names.begin(), TN_names.end(), this->names[i]);
        cytnx_error_msg(
          it == end(TN_names),
          "[ERROR][Network][Contract--planning] TN: <%s> defined but is not used in ORDER line\n",
          this->names[i].c_str());
        TN_names.erase(it);
      }
      if (TN_names.size() != 0) {
        cout << "[ERROR] Following TNs appeared in ORDER line, but is not defined." << endl;
        for (int i = 0; i < TN_names.size(); i++) {
          cout << "        " << TN_names[i] << endl;
        }
        cytnx_error_msg(true, "%s", "\n");
      }

    }  // check all RN.

    // checking label matching:
    map<string, cytnx_int64> labelcnt;
    for (int i = 0; i < this->names.size(); i++) {
      for (int j = 0; j < this->label_arr[i].size(); j++) {
        if (labelcnt.find(this->label_arr[i][j]) == labelcnt.end())
          labelcnt[this->label_arr[i][j]] = 1;
        else
          labelcnt[this->label_arr[i][j]] += 1;
      }
    }
    vector<string> expected_TOUT;
    for (map<string, cytnx_int64>::iterator it = labelcnt.begin(); it != labelcnt.end(); ++it) {
      if (it->second == 1) expected_TOUT.push_back(it->first);
    }
    bool err = false;
    if (expected_TOUT.size() != TOUT_labels.size()) {
      cout << expected_TOUT.size() << endl;
      err = true;
    }
    vector<string> itrsct = vec_intersect(expected_TOUT, this->TOUT_labels);
    if (itrsct.size() != expected_TOUT.size()) {
      err = true;
    }

    if (err) {
      cout << "[ERROR][Network][Contract--planning] The TOUT contains labels that does not match "
              "with the delcartion from TNs.\n";
      cout << "  > The reduced labels [rank:" << expected_TOUT.size() << "] should be:";
      for (int i = 0; i < expected_TOUT.size(); i++) cout << expected_TOUT[i] << " ";
      cout << endl;
      cout << "  > The TOUT [rank" << TOUT_labels.size() << "] specified is:";
      for (int i = 0; i < TOUT_labels.size(); i++) cout << TOUT_labels[i] << " ";
      cout << endl;
      cytnx_error_msg(true, "%s", "\n");
    }

    // put tensor:
    for (int i = 0; i < utensors.size(); i++) this->tensors[i] = utensors[i];
  }

  void RegularNetwork::FromString(const vector<string> &contents) {
    this->clear();

    string line;
    vector<string> tmpvs;
    bool isORDER_exist = false;

    for (int i = 0; i < contents.size(); i++) {
      line = contents[i];
      line = str_strip(line, "\n");
      line = str_strip(line);
      if (line.length() == 0) continue;  // blank line
      if (line.at(0) == '#') continue;  // comment whole line.

      // remove any comment at eol :
      line = str_split(line, true, "#")[0];  // remove comment on end.

      // A valid line should contain ':':
      cytnx_error_msg(
        line.find_first_of(":") == string::npos,
        "[ERROR][Network][Fromfile] invalid network description at line: %d. should contain \':\'",
        i);

      tmpvs = str_split(line, false, ":");  // note that checking empty string!
      cytnx_error_msg(tmpvs.size() != 2,
                      "[ERROR][Network] invalid line in network file at line: %d. \n", i);

      string name = str_strip(tmpvs[0]);
      string content = str_strip(tmpvs[1]);

      // check if name contain invalid keyword or not assigned.
      cytnx_error_msg(name.length() == 0,
                      "[ERROR][Network][Fromfile] invalid tensor name at line: %d\n", i);
      cytnx_error_msg(name.find_first_of(" ;,") != string::npos,
                      "[ERROR] invalid Tensor name at line %d\n", i);

      // dispatch
      if (name == "ORDER") {
        if (content.length()) {
          // cut the line into tokens,
          // and leave it to process by CtTree after read all lines.
          this->order_line = content;
          _parse_ORDER_line_(this->ORDER_tokens, content, i);
          isORDER_exist = true;
        }
      } else if (name == "TOUT") {
        // if content has length, then pass to process.
        if (content.length()) {
          // this is an internal function that is defined in this cpp file.
          _parse_TOUT_line_(this->TOUT_labels, this->TOUT_iBondNum, content, i);
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
                        i);

        this->name2pos[name] = names.size() - 1;  // register
        // cout << name << "|" << names.size() - 1 << endl;
        this->label_arr.push_back(vector<string>());
        cytnx_uint64 tmp_iBN;
        // this is an internal function that is defined in this cpp file.
        _parse_TN_line_(this->label_arr.back(), tmp_iBN, content, i);
        this->iBondNums.push_back(tmp_iBN);
      }

    }  //

    // cytnx_error_msg(lnum>=MAXLINES,"[ERROR][Network][Fromfile] network file exceed the maxinum
    // allowed lines, MAXLINES=1024%s","\n");

    cytnx_error_msg(
      this->names.size() < 2,
      "[ERROR][Network][Fromfile] invalid network file. Should have at least 2 tensors defined.%s",
      "\n");

    this->tensors.resize(this->names.size());
    this->CtTree.base_nodes.resize(this->names.size());

    // checking if all TN are set in ORDER.
    if (isORDER_exist) {
      vector<string> TN_names;
      _extract_TNs_from_ORDER_(TN_names, this->ORDER_tokens);
      for (int i = 0; i < this->names.size(); i++) {
        auto it = find(TN_names.begin(), TN_names.end(), this->names[i]);
        cytnx_error_msg(
          it == end(TN_names),
          "[ERROR][Network][Fromfile] TN: <%s> defined but is not used in ORDER line\n",
          this->names[i].c_str());
        TN_names.erase(it);
      }
      if (TN_names.size() != 0) {
        cout << "[ERROR] Following TNs appeared in ORDER line, but is not defined." << endl;
        for (int i = 0; i < TN_names.size(); i++) {
          cout << "        " << TN_names[i] << endl;
        }
        cytnx_error_msg(true, "%s", "\n");
      }
    }  // check all RN.

    // checking label matching:
    map<string, cytnx_int64> labelcnt;
    for (int i = 0; i < this->names.size(); i++) {
      for (int j = 0; j < this->label_arr[i].size(); j++) {
        if (labelcnt.find(this->label_arr[i][j]) == labelcnt.end())
          labelcnt[this->label_arr[i][j]] = 1;
        else
          labelcnt[this->label_arr[i][j]] += 1;
      }
    }
    vector<string> expected_TOUT;
    for (map<string, cytnx_int64>::iterator it = labelcnt.begin(); it != labelcnt.end(); ++it) {
      if (it->second == 1) expected_TOUT.push_back(it->first);
    }
    bool err = false;
    if (expected_TOUT.size() != TOUT_labels.size()) {
      err = true;
    }
    vector<string> itrsct = vec_intersect(expected_TOUT, this->TOUT_labels);
    if (itrsct.size() != expected_TOUT.size()) {
      err = true;
    }

    if (err) {
      cout << "[ERROR][Network][Fromfile] The TOUT contains labels that does not match with the "
              "delcartion from TNs.\n";
      cout << "  > The reduced labels [rank:" << expected_TOUT.size() << "] should be:";
      for (int i = 0; i < expected_TOUT.size(); i++) cout << expected_TOUT[i] << " ";
      cout << endl;
      cout << "  > The TOUT [rank" << TOUT_labels.size() << "] specified is:";
      for (int i = 0; i < TOUT_labels.size(); i++) cout << TOUT_labels[i] << " ";
      cout << endl;
      cytnx_error_msg(true, "%s", "\n");
    }

    // maintain TOUT leg position
    TOUT_pos = vector<pair<int, int>>();
    for (int i = 0; i < TOUT_labels.size(); i++) {
      for (int j = 0; j < label_arr.size(); j++) {
        vector<string>::iterator it;
        it = find(this->label_arr[j].begin(), this->label_arr[j].end(), TOUT_labels[i]);
        if (it != this->label_arr[j].end()) {
          TOUT_pos.push_back(make_pair(j, distance(label_arr[j].begin(), it)));
          break;
        }
      }
    }

    // get int_label
    std::map<std::string, cytnx_int64> labelmap = std::map<std::string, cytnx_int64>();
    this->int_modes = std::vector<std::vector<cytnx_int64>>(this->label_arr.size());
    this->int_out_mode = std::vector<cytnx_int64>(this->TOUT_labels.size());
    cytnx_int64 label_int = 0;
    for (size_t i = 0; i < this->label_arr.size(); i++) {
      this->int_modes[i] = std::vector<cytnx_int64>(this->label_arr[i].size());
      for (size_t j = 0; j < this->label_arr[i].size(); j++) {
        labelmap.insert(std::pair<std::string, cytnx_int64>(this->label_arr[i][j], label_int));
        this->int_modes[i][j] = labelmap[this->label_arr[i][j]];
        label_int += 1;
      }
    }
    for (size_t i = 0; i < TOUT_labels.size(); i++) {
      this->int_out_mode[i] = labelmap[this->TOUT_labels[i]];
    }

  #ifdef UNI_GPU
    #ifdef UNI_CUQUANTUM
    this->optimizerInfo = nullptr;
    #endif
  #endif

    vector<string> names;
    for (int i = 0; i < this->names.size(); i++) {
      names.push_back(this->names[i]);
      CtTree.base_nodes[i].name = this->names[i];
    }
    if (ORDER_tokens.size() != 0) {
      CtTree.build_contraction_tree_by_tokens(this->name2pos, ORDER_tokens);
    } else {
      CtTree.build_default_contraction_tree();
    }
    // this->einsum_path = CtTree_to_eisumpath(CtTree, names);
  }

  void RegularNetwork::Fromfile(const string &fname) {
    const cytnx_uint64 MAXLINES = 1024;

    // empty all
    // this->clear();

    // open file
    ifstream infile;
    infile.open(fname.c_str());
    if (!(infile.is_open())) {
      cytnx_error_msg(true, "[Network] Error in opening file \'", fname.c_str(), "\'.\n");
    }
    filename = fname;

    string line;
    cytnx_uint64 lnum = 0;

    vector<string> contents;

    // read each line:
    while (lnum < MAXLINES) {
      lnum++;
      getline(infile, line);
      contents.push_back(line);
      if (infile.eof()) break;

    }  // end readlines

    bool iseof = infile.eof();
    infile.close();

    cytnx_error_msg(
      !iseof,
      "[ERROR][Network][Fromfile] network file exceed the maxinum allowed lines, MAXLINES=1024%s",
      "\n");

    this->FromString(contents);
  }

  void RegularNetwork::PutUniTensors(const vector<string> &names,
                                     const vector<UniTensor> &utensors) {
    cytnx_error_msg(names.size() != utensors.size(),
                    "[ERROR][RegularNetwork][PutUniTensors] total number of names does not match "
                    "number of input UniTensors.%s",
                    "\n");
    for (int i = 0; i < names.size(); i++) {
      this->PutUniTensor(names[i], utensors[i]);
    }
  }

  void RegularNetwork::PutUniTensor(const cytnx_uint64 &idx, const UniTensor &utensor) {
    cytnx_error_msg(idx >= this->CtTree.base_nodes.size(),
                    "[ERROR][RegularNetwork][PutUniTensor] index=%d out of range.\n", idx);

    // check shape:
    cytnx_error_msg(this->label_arr[idx].size() != utensor.rank(),
                    "[ERROR][RegularNetwork][PutUniTensor] tensor name: [%s], the rank of input "
                    "UniTensor does not match the definition in network file.\n",
                    this->names[idx].c_str());

    this->tensors[idx] = utensor;
  }

  void RegularNetwork::RmUniTensor(const cytnx_uint64 &idx) {
    cytnx_error_msg(idx >= this->CtTree.base_nodes.size(),
                    "[ERROR][RegularNetwork][RmUniTensor] index=%d out of range.\n", idx);

    this->tensors[idx] = UniTensor();
  }
  void RegularNetwork::RmUniTensor(const string &name) {
    cytnx_uint64 idx;
    try {
      idx = this->name2pos.at(name);
    } catch (out_of_range) {
      cytnx_error_msg(true,
                      "[ERROR][RegularNetwork][RmUniTensor] cannot find the tensor name: [%s] in "
                      "current network.\n",
                      name.c_str());
    }

    this->RmUniTensor(idx);
  }
  void RegularNetwork::RmUniTensors(const vector<string> &names) {
    for (int i = 0; i < names.size(); i++) {
      this->RmUniTensor(names[i]);
    }
  }

  void RegularNetwork::Savefile(const string &fname) {
    cytnx_error_msg(
      this->label_arr.size() == 0,
      "[ERROR][RegularNetwork][Savefile] cannot save empty network to network file!%s", "\n");

    fstream fo;
    fo.open(fname + ".net", ios::out | ios::trunc);
    if (!fo.is_open()) {
      cytnx_error_msg(true, "[ERROR][RegularNetwork][Savefile] cannot open/create file:%s\n",
                      fname.c_str());
    }

    for (int i = 0; i < this->label_arr.size(); i++) {
      fo << this->names[i] << " : ";
      // if (this->iBondNums[i] == 0) fo << ";";

      for (int j = 0; j < this->label_arr[i].size(); j++) {
        fo << this->label_arr[i][j];

        // if (j + 1 == this->iBondNums[i])
        //   fo << ";";
        if (j != this->label_arr[i].size() - 1) fo << ",";

        if (j == this->label_arr[i].size() - 1) fo << "\n";
      }
    }

    fo << "TOUT : ";
    for (int i = 0; i < TOUT_iBondNum; i++) {
      fo << this->TOUT_labels[i];
      if (i != this->TOUT_iBondNum - 1) {
        fo << ",";
      }
    }
    if (this->TOUT_labels.size() != 0) fo << ";";
    for (int i = TOUT_iBondNum; i < this->TOUT_labels.size(); i++) {
      fo << this->TOUT_labels[i];
      if (i != this->TOUT_labels.size() - 1) {
        fo << ",";
      }
    }
    fo << "\n";

    if (ORDER_tokens.size() != 0) {
      fo << "ORDER : ";
      for (int i = 0; i < ORDER_tokens.size(); i++) {
        fo << ORDER_tokens[i];
      }
      fo << endl;
    }

    fo.close();
  }

  void RegularNetwork::PutUniTensor(const string &name, const UniTensor &utensor) {
    cytnx_uint64 idx;
    try {
      idx = this->name2pos.at(name);
    } catch (out_of_range) {
      cytnx_error_msg(true,
                      "[ERROR][RegularNetwork][PutUniTensor] cannot find the tensor name: [%s] in "
                      "current network.\n",
                      name.c_str());
    }

    this->PutUniTensor(idx, utensor);
  }

  void RegularNetwork::PrintNet(ostream &os) {
    string status;
    os << "==== Network ====" << endl;
    if (this->tensors.size() == 0) {
      os << "      Empty      " << endl;
    } else {
      for (cytnx_uint64 i = 0; i < this->tensors.size(); i++) {
        if (this->tensors[i].uten_type() != UTenType.Void)
          status = "o";
        else
          status = "x";
        os << "[" << status.c_str() << "] " << this->names[i].c_str() << " : ";
        // printf("[%s] %s : ",status.c_str(), this->names[i].c_str());

        for (cytnx_int64 j = 0; j < this->iBondNums[i]; j++) {
          os << this->label_arr[i][j] << " ";
          // printf("%d ",this->label_arr[i][j]);
        }
        // os << "; ";
        // printf("%s","; ");
        for (cytnx_int64 j = this->iBondNums[i]; j < this->label_arr[i].size(); j++) {
          os << this->label_arr[i][j] << " ";
          // printf("%d ",this->label_arr[i][j]);
        }
        os << endl;
      }

      os << "TOUT : ";
      for (cytnx_uint64 i = 0; i < TOUT_iBondNum; i++) {
        os << this->TOUT_labels[i] << " ";
        // printf("%d ",this->TOUT_labels[i]);
      }
      os << "; ";
      // printf("%s","; ");
      for (cytnx_int64 j = this->TOUT_iBondNum; j < this->TOUT_labels.size(); j++) {
        os << this->TOUT_labels[j] << " ";
        // printf("%d ",this->TOUT_labels[j]);
      }
      os << endl;
      os << "ORDER : ";
      for (cytnx_int64 i = 0; i < this->ORDER_tokens.size(); i++) {
        os << this->ORDER_tokens[i];
      }
      os << endl;
      os << "=================" << endl;
    }
  }

  string RegularNetwork::getOrder() {
    if (this->order_line != "") {
      return this->order_line;
    } else {
      return "No optimal or user specified order found.";
    }
  }

  void RegularNetwork::setOrder(const bool &optimal, const string &contract_order /*default ""*/) {
    cytnx_warning_msg(optimal && (contract_order != ""),
                      "[WARNING][setOrder][RegularNetwork] Setting Optimal = true while specifying "
                      "the order, will find the optimal order instead."
                      "to use the desired order please set Optimal = false.%s",
                      "\n");
    cytnx_warning_msg((!optimal) && (contract_order == ""),
                      "[WARNING][setOrder][RegularNetwork] Setting Optimal = false while not "
                      "specifying the order string, will use default contraciton order instrad.%s",
                      "\n");
    this->ORDER_tokens.clear();
    if (optimal) {
      check(this->tensors, this->names);

      if (this->tensors[0].device() == -1) {
        string Optim_ORDERline = this->getOptimalOrder();
        this->order_line = Optim_ORDERline;
        _parse_ORDER_line_(ORDER_tokens, Optim_ORDERline, 999999);
      } else {
  #ifdef UNI_GPU
    #ifdef UNI_CUQUANTUM
        if (this->tensors[0].uten_type() != UTenType.Dense) {
          cytnx_error_msg(true, "[ERROR][setOrder][RegularNetwork] Error,%s",
                          "Sparse or Block type UniTensor network optimization is not support.\n");
        } else {
          vector<cytnx_uint64> out_shape;
          for (int i = 0; i < this->TOUT_labels.size(); i++) {
            out_shape.push_back(this->tensors[TOUT_pos[i].first].shape()[TOUT_pos[i].second]);
          }
          cutensornet cutn;
          cutn.setDevice(this->tensors[0].device());
          cutn.createStream();
          cutn.createHandle();
          cutn.parseLabels(this->int_out_mode, this->int_modes);
          cutn.set_output_extents(out_shape);
          cutn.set_extents(this->tensors);
          cutn.checkVersion();
          this->descNet = cutn.createNetworkDescriptor();
          cutn.getWorkspacelimit();
          this->optimizerInfo = cutn.findOptimalOrder();

          // Get contraction path
          vector<pair<cytnx_int64, cytnx_int64>> path = cutn.getContractionPath();
          cutn.freeHandle();

          vector<string> names;
          for (int i = 0; i < this->names.size(); i++) {
            names.push_back(this->names[i]);
          }
          this->einsum_path = path;
          this->order_line = einsumpath_to_string(path, names);
        }
    #else
        // cytnx_error_msg(true, "[ERROR][setOrder][RegularNetwork] fatal error,%s",
        //                 "try to call the gpu section for finding optimal contraction order
        //                 without " "CUQUANTUM support.\n");
        string Optim_ORDERline = this->getOptimalOrder();
        this->order_line = Optim_ORDERline;
        _parse_ORDER_line_(ORDER_tokens, Optim_ORDERline, 999999);
    #endif

  #else
        cytnx_error_msg(true, "[ERROR][setOrder][RegularNetwork] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
  #endif
      }

    } else {
      this->order_line = contract_order;
      if (contract_order != "") {
        _parse_ORDER_line_(ORDER_tokens, contract_order, 999999);
        CtTree.build_contraction_tree_by_tokens(this->name2pos, ORDER_tokens);
        // this->einsum_path = CtTree_to_eisumpath(CtTree, names);
      } else {
        CtTree.build_default_contraction_tree();
      }
    }
  }

  string RegularNetwork::getOptimalOrder() {
    // Creat a SearchTree to search for optim contraction order.
    SearchTree Stree;
    Stree.base_nodes.resize(this->tensors.size());
    for (cytnx_uint64 t = 0; t < this->tensors.size(); t++) {
      Stree.base_nodes[t].from_utensor(this->tensors[t]);  // create psudotensors from base tensors
      // Stree.base_nodes[t].from_utensor(CtTree.base_nodes[t].utensor);
      Stree.base_nodes[t].accu_str = this->names[t];
    }
    Stree.search_order();
    return Stree.nodes_container.back()[0].accu_str;
  }

  UniTensor RegularNetwork::Launch() {
    check(this->tensors, this->names);

    int tn_device = this->tensors[0].device();

    if (tn_device == -1) {
      // cpu workflow

      for (cytnx_uint64 idx = 0; idx < this->tensors.size(); idx++) {
        this->CtTree.base_nodes[idx].utensor =
          this->tensors[idx].relabels(this->label_arr[idx]);  // this conflict
        this->CtTree.base_nodes[idx].is_assigned = true;
      }
      // 1.5 contraction order:
      if (ORDER_tokens.size() != 0) {
        // *set by user or optimally found
        CtTree.build_contraction_tree_by_tokens(this->name2pos, ORDER_tokens);
      } else {
        CtTree.build_default_contraction_tree();
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
            // cout << "L,R::\n";
            // root->left->utensor.print_diagram(1);
            // root->right->utensor.print_diagram(1);
            root->utensor = Contract(root->left->utensor, root->right->utensor);
            // root->left->utensor.print_diagram(); root->right->utensor.print_diagram();
            // root->utensor.print_diagram(); root->utensor.set_name(root->left->utensor.name() +
            // root->right->utensor.name());
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
      // cout << out << endl;
      // out.print_diagram();

      // 4. reset nodes:
      this->CtTree.reset_nodes();

      // //5. reset back the original labels:
      // for(cytnx_uint64 i=0;i<this->tensors.size();i++){
      //     this->tensors[i].set_labels(old_labels[i]);
      // }

      // 6. permute accroding to pre-set labels:
      if (TOUT_labels.size()) {
        out.permute_(TOUT_labels, TOUT_iBondNum);
      }
      // UniTensor out;
      return out;

    } else {
      // gpu workflow
  #ifdef UNI_GPU
    #ifdef UNI_CUQUANTUM
      if (this->tensors[0].uten_type() != UTenType.Dense) {
        cytnx_error_msg(true, "[ERROR][Launch][RegularNetwork] Error,%s",
                        "Sparse or Block type UniTensor network contraction is not support.\n");
        return UniTensor();
      } else {
        vector<cytnx_uint64> out_shape;
        for (int i = 0; i < this->TOUT_labels.size(); i++) {
          out_shape.push_back(this->tensors[TOUT_pos[i].first].shape()[TOUT_pos[i].second]);
        }
        UniTensor out =
          UniTensor(zeros(out_shape, this->tensors[0].dtype(), this->tensors[0].device()));
        cutensornet cutn;
        cutn.setDevice(this->tensors[0].device());
        cutn.createStream();
        cutn.createHandle();
        cutn.parseLabels(this->int_out_mode, this->int_modes);
        cutn.set_output_extents(out_shape);
        cutn.set_extents(this->tensors);
        this->descNet = cutn.createNetworkDescriptor();
        cutn.setOutputMem(out);
        cutn.setInputMem(this->tensors);
        // cutn.setNetworkDescriptor(this->descNet);
        if (this->optimizerInfo != nullptr) {
          cutn.setOptimizerInfo(this->optimizerInfo);
        } else {
          this->optimizerInfo = cutn.createOptimizerInfo();
        }
        cutn.setContractionPath(einsum_path);

        // cutn.getContractionPath();

        cutn.createWorkspaceDescriptor();
        cutn.initializePlan();
        cutn.autotune();
        cutn.executeContraction();
        cutn.freePlan();
        cutn.freeWorkspaceDescriptor();
        cutn.freeHandle();

        return out;
      }
    #else
      // 1.5 contraction order:
      if (ORDER_tokens.size() != 0) {
        // *set by user or optimally found
        CtTree.build_contraction_tree_by_tokens(this->name2pos, ORDER_tokens);
      } else {
        CtTree.build_default_contraction_tree();
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
          }
          root = nullptr;
        }
      } while (!stk.empty());
      // 3. get result:
      UniTensor out = this->CtTree.nodes_container.back().utensor;
      // 4. reset nodes:
      this->CtTree.reset_nodes();
      // 6. permute accroding to pre-set labels:
      if (TOUT_labels.size()) {
        out.permute_(TOUT_labels, TOUT_iBondNum);
      }
      return out;
    #endif

  #else
      cytnx_error_msg(true, "[ERROR][Launch][RegularNetwork] fatal error,%s",
                      "try to call the gpu section without CUDA support.\n");
      return UniTensor();
  #endif
    }
  }

  void RegularNetwork::construct(const vector<string> &alias, const vector<vector<string>> &labels,
                                 const vector<string> &outlabel, const cytnx_int64 &outrk,
                                 const string &order, const bool optim) {
    this->clear();
    for (int i = 0; i < alias.size(); i++) {
      this->names.push_back(alias[i]);
      this->name2pos[alias[i]] = names.size() - 1;  // register
      cytnx_uint64 tmp_iBN = labels[i].size();
      // this is an internal function that is defined in this cpp file.
      this->label_arr.push_back(labels[i]);
      this->iBondNums.push_back(tmp_iBN);
    }
    this->TOUT_labels = outlabel;
    this->TOUT_iBondNum = outlabel.size() - outrk;

    if (order.length()) {
      this->order_line = order;
      // checking if all TN are set in ORDER.
      _parse_ORDER_line_(this->ORDER_tokens, order, 0);
      vector<string> TN_names;
      _extract_TNs_from_ORDER_(TN_names, this->ORDER_tokens);
      for (int i = 0; i < this->names.size(); i++) {
        auto it = find(TN_names.begin(), TN_names.end(), this->names[i]);
        cytnx_error_msg(
          it == end(TN_names),
          "[ERROR][Network][Fromfile] TN: <%s> defined but is not used in ORDER line\n",
          this->names[i].c_str());
        TN_names.erase(it);
      }
      if (TN_names.size() != 0) {
        cout << "[ERROR] Following TNs appeared in ORDER line, but is not defined." << endl;
        for (int i = 0; i < TN_names.size(); i++) {
          cout << "        " << TN_names[i] << endl;
        }
        cytnx_error_msg(true, "%s", "\n");
      }
    }  // check all RN.

    cytnx_error_msg(
      this->names.size() < 2,
      "[ERROR][Network][Fromfile] invalid network file. Should have at least 2 tensors defined.%s",
      "\n");

    this->tensors.resize(this->names.size());
    this->CtTree.base_nodes.resize(this->names.size());

    // checking label matching:
    map<string, cytnx_int64> labelcnt;
    for (int i = 0; i < this->names.size(); i++) {
      for (int j = 0; j < this->label_arr[i].size(); j++) {
        if (labelcnt.find(this->label_arr[i][j]) == labelcnt.end())
          labelcnt[this->label_arr[i][j]] = 1;
        else
          labelcnt[this->label_arr[i][j]] += 1;
      }
    }
    vector<string> expected_TOUT;

    for (map<string, cytnx_int64>::iterator it = labelcnt.begin(); it != labelcnt.end(); ++it) {
      if (it->second == 1) expected_TOUT.push_back(it->first);
    }
    // cout<<this->TOUT_labels.size();
    if (this->TOUT_labels.size() == 0) {
      this->TOUT_labels = expected_TOUT;
    } else {
      bool err = false;
      if (expected_TOUT.size() != TOUT_labels.size()) {
        err = true;
      }
      vector<string> itrsct = vec_intersect(expected_TOUT, this->TOUT_labels);
      if (itrsct.size() != expected_TOUT.size()) {
        err = true;
      }
      if (err) {
        cout << "[ERROR][Network][Fromfile] The TOUT contains labels that does not match with the "
                "delcartion from TNs.\n";
        cout << "  > The reduced labels [rank:" << expected_TOUT.size() << "] should be:";
        for (int i = 0; i < expected_TOUT.size(); i++) cout << expected_TOUT[i] << " ";
        cout << endl;
        cout << "  > The TOUT [rank" << TOUT_labels.size() << "] specified is:";
        for (int i = 0; i < TOUT_labels.size(); i++) cout << TOUT_labels[i] << " ";
        cout << endl;
        cytnx_error_msg(true, "%s", "\n");
      }
    }

    // maintain TOUT leg position
    TOUT_pos = vector<pair<int, int>>();
    for (int i = 0; i < TOUT_labels.size(); i++) {
      for (int j = 0; j < label_arr.size(); j++) {
        vector<string>::iterator it;
        it = find(this->label_arr[j].begin(), this->label_arr[j].end(), TOUT_labels[i]);
        if (it != this->label_arr[j].end()) {
          TOUT_pos.push_back(make_pair(j, distance(label_arr[j].begin(), it)));
          break;
        }
      }
    }

    // get int_label
    std::map<std::string, cytnx_int64> labelmap = std::map<std::string, cytnx_int64>();
    this->int_modes = std::vector<std::vector<cytnx_int64>>(this->label_arr.size());
    this->int_out_mode = std::vector<cytnx_int64>(this->TOUT_labels.size());
    cytnx_int64 label_int = 0;
    for (size_t i = 0; i < this->label_arr.size(); i++) {
      this->int_modes[i] = std::vector<cytnx_int64>(this->label_arr[i].size());
      for (size_t j = 0; j < this->label_arr[i].size(); j++) {
        labelmap.insert(std::pair<std::string, cytnx_int64>(this->label_arr[i][j], label_int));
        this->int_modes[i][j] = labelmap[this->label_arr[i][j]];
        label_int += 1;
      }
    }
    for (size_t i = 0; i < TOUT_labels.size(); i++) {
      this->int_out_mode[i] = labelmap[this->TOUT_labels[i]];
    }

  #ifdef UNI_GPU
    #ifdef UNI_CUQUANTUM
    this->optimizerInfo = nullptr;
    #endif
  #endif

    vector<string> names;
    for (int i = 0; i < this->names.size(); i++) {
      names.push_back(this->names[i]);
      CtTree.base_nodes[i].name = this->names[i];
    }
    if (ORDER_tokens.size() != 0) {
      CtTree.build_contraction_tree_by_tokens(this->name2pos, ORDER_tokens);
    } else {
      CtTree.build_default_contraction_tree();
    }
    // this->einsum_path = CtTree_to_eisumpath(CtTree, names);
  }  // end construct

}  // namespace cytnx
#endif
