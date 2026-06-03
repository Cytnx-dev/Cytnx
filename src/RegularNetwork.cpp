#include <algorithm>
#include <iostream>
#include <stack>
#include <typeinfo>

#include "Device.hpp"
#include "Generator.hpp"
#include "Network.hpp"
#include "search_tree.hpp"

#ifdef BACKEND_TORCH
#else
  #include "utils/cutensornet.hpp"

namespace cytnx {

  namespace {
    std::vector<std::string> ParseOrderLineInternal(const std::string &line,
                                                    const cytnx_uint64 &line_num) {
      cytnx_error_msg((line.find_first_of("\t;\n:") != std::string::npos),
                      "[ERROR][Network][Fromfile] line:%d invalid ORDER line format.%s", line_num,
                      "\n");
      cytnx_error_msg((line.find_first_of("(),") == std::string::npos),
                      "[ERROR][Network][Fromfile] line:%d invalid ORDER line format.%s", line_num,
                      " tensors should be seperate by delimiter \',\' (comma), and/or wrapped with "
                      "\'(\' and \')\'");

      // check mismatch:
      std::size_t lbrac_n = std::count(line.begin(), line.end(), '(');
      std::size_t rbrac_n = std::count(line.begin(), line.end(), ')');
      cytnx_error_msg(lbrac_n != rbrac_n, "[ERROR][Network][Fromfile] parentheses mismatch.%s",
                      "\n");

      // slice the line into pieces by parentheses and comma
      std::vector<std::string> tokens = str_findall(line, "(),");

      cytnx_error_msg(tokens.size() == 0,
                      "[ERROR][Network][Fromfile] line:%d invalid ORDER line.%s", line_num, "\n");
      return tokens;
    }

    std::vector<std::string> ParseToutLineInternal(cytnx_uint64 &TOUT_iBondNum,
                                                   const std::string &line,
                                                   const cytnx_uint64 &line_num) {
      std::vector<std::string> labels;

      std::vector<std::string> tmp = str_split(line, false, ";");
      // cytnx_error_msg(tmp.size() != 2, "[ERROR][Network][Fromfile] line:%d %s\n", line_num,
      //                 "Invalid TOUT line");
      if (tmp.size() == 1) {
        // no ; provided
        std::vector<std::string> tout_lbls = str_split(line, false, ",");
        cytnx_uint64 default_rowrank = tout_lbls.size() / 2;
        // handle col-space label
        for (cytnx_uint64 i = 0; i < default_rowrank; i++) {
          std::string tmp_ = str_strip(tout_lbls[i]);
          cytnx_error_msg(tmp_.length() == 0,
                          "[ERROR][Network][Fromfile] line:%d Invalid labels for TOUT line.%s",
                          line_num, "\n");
          labels.push_back(tmp_);
        }
        TOUT_iBondNum = labels.size();
        // handle row-space label
        for (cytnx_uint64 i = default_rowrank; i < tout_lbls.size(); i++) {
          std::string tmp_ = str_strip(tout_lbls[i]);
          cytnx_error_msg(tmp_.length() == 0,
                          "[ERROR][Network][Fromfile] line:%d Invalid labels for TOUT line.%s",
                          line_num, "\n");
          labels.push_back(tmp_);
        }
      } else if (tmp.size() == 2) {
        // one ;
        // handle col-space label
        std::vector<std::string> ket_labels = str_split(tmp[0], false, ",");
        if (ket_labels.size() == 1)
          if (ket_labels[0].length() == 0) ket_labels.clear();
        for (cytnx_uint64 i = 0; i < ket_labels.size(); i++) {
          std::string tmp = str_strip(ket_labels[i]);
          cytnx_error_msg(tmp.length() == 0,
                          "[ERROR][Network][Fromfile] line:%d Invalid labels for TOUT line.%s",
                          line_num, "\n");
          labels.push_back(tmp);
        }
        TOUT_iBondNum = labels.size();
        // handle row-space label
        std::vector<std::string> bra_labels = str_split(tmp[1], false, ",");
        if (bra_labels.size() == 1)
          if (bra_labels[0].length() == 0) bra_labels.clear();
        for (cytnx_uint64 i = 0; i < bra_labels.size(); i++) {
          std::string tmp = str_strip(bra_labels[i]);
          cytnx_error_msg(tmp.length() == 0,
                          "[ERROR][Network][Fromfile] line:%d Invalid labels for TOUT line.%s",
                          line_num, "\n");
          labels.push_back(tmp);
        }
      } else if (tmp.size() > 2) {
        // more than one ;
        cytnx_error_msg(true, "[ERROR][Network] line:%d Invalid TOUT line.%s", line_num, "\n");
      }
      return labels;
    }

    std::vector<std::string> ParseTnLineInternal(const std::string &line,
                                                 const cytnx_uint64 &line_num) {
      std::vector<std::string> labels;

      std::vector<std::string> alllabels = str_split(line, false, ",");
      if (alllabels.size() == 1)
        if (alllabels[0].length() == 0) alllabels.clear();
      for (cytnx_uint64 i = 0; i < alllabels.size(); i++) {
        std::string tmp = str_strip(alllabels[i]);
        cytnx_error_msg(tmp.length() == 0,
                        "[ERROR][Network][Fromfile] line:%d Invalid labels for TN line.%s",
                        line_num, "\n");
        // cytnx_error_msg((tmp.find_first_not_of("0123456789-") != std::string::npos),
        //                 "[ERROR][Network][Fromfile] line:%d %s\n", line_num,
        //                 "Invalid TN line. label contain non integer.");
        labels.push_back(tmp);
      }

      cytnx_error_msg(labels.size() == 0, "[ERROR][Network][Fromfile] line:%d %s\n", line_num,
                      "Invalid TN line. no label present in this line, which is invalid.%s", "\n");
      return labels;
    }

    std::vector<std::string> ExtractTnsFromOrderInternal(const std::vector<std::string> &tokens) {
      std::vector<std::string> TN_names;
      for (cytnx_uint64 i = 0; i < tokens.size(); i++) {
        std::string tok = str_strip(tokens[i]);  // remove space.
        if (tok.length() == 0) continue;
        if ((tok != "(") && (tok != ")") && (tok != ",")) {
          TN_names.push_back(tok);
        }
      }
      return TN_names;
    }

    std::string EinsumpathToStringInternal(std::vector<std::pair<cytnx_int64, cytnx_int64>> path,
                                           std::vector<std::string> tns) {
      std::string res;
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

    std::vector<std::pair<cytnx_int64, cytnx_int64>> CtTreeToEinsumpathInternal(
      ContractionTree CtTree, std::vector<std::string> tns) {
      std::vector<std::pair<cytnx_int64, cytnx_int64>> path;
      std::stack<std::shared_ptr<Node>> stk;
      std::shared_ptr<Node> root = CtTree.nodes_container.back();
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
            auto it = std::find(tns.begin(), tns.end(), root->left->name);
            int id1 = it - tns.begin();
            it = std::find(tns.begin(), tns.end(), root->right->name);
            int id2 = it - tns.begin();
            tns.erase(tns.begin() + id1);
            if (id1 > id2)
              tns.erase(tns.begin() + id2);
            else
              tns.erase(tns.begin() + id2 - 1);
            tns.push_back("(" + root->left->name + "," + root->right->name + ")");
            root->name = "(" + root->left->name + "," + root->right->name + ")";
            path.push_back(std::pair<cytnx_int64, cytnx_int64>(id1, id2));
          }
          root = nullptr;
        }
      } while (!stk.empty());
      return path;
    }

    void CheckInternal(std::vector<UniTensor> &tns, std::vector<std::string> &tn_names) {
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
                        "[ERROR][Launch][RegularNetwork] Cannot find optimal order/launch with "
                        "tensors on different devices, tensor "
                        "at [0] is on device %d while tensor at [%d] in on device %d. %s",
                        tn_device, i, tns[i].device(), "\n");
        cytnx_error_msg(tns[i].uten_type() != utentype,
                        "[ERROR][Launch][RegularNetwork] Cannot find optimal order/launch with "
                        "tensors of different unitensor types, tensor "
                        "at [0] is uten_type %d while tensor at [%d] in uten_type %d. %s",
                        utentype, i, tns[i].uten_type(), "\n");
      }
    }
  }  // namespace

  void RegularNetwork::Contract_plan(const std::vector<UniTensor> &utensors,
                                     const std::string &Tout, const std::vector<std::string> &alias,
                                     const std::string &contract_order) {
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
      this->ORDER_tokens = ParseOrderLineInternal(contract_order, 0);
      isORDER_exist = true;
    }
    if (Tout.length()) {
      // TOUT assigned
      this->TOUT_labels = ParseToutLineInternal(this->TOUT_iBondNum, Tout, 0);
    }

    // assign input tensors into slots:
    std::string name;
    for (unsigned int i = 0; i < utensors.size(); i++) {
      if (alias.size()) {
        this->names.push_back(alias[i]);
        name = alias[i];
      } else {
        if (utensors[i].name().length()) {
          name = utensors[i].name() + "_usr";
        } else {
          name = "uname_T" + std::to_string(i);
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
      this->label_arr.push_back(std::vector<std::string>());
      cytnx_uint64 tmp_iBN;
      // this is an internal function that is defined in this cpp file.
      this->label_arr.back() = utensors[i].labels();
      this->iBondNums.push_back(utensors[i].rowrank());
    }  // traversal input tensor list

    this->tensors.resize(this->names.size());
    this->CtTree.base_nodes.resize(this->names.size());

    // checking if all TN are set in ORDER.
    //  only alias assigned will activate order
    if (isORDER_exist) {
      std::vector<std::string> TN_names;  // this should be integer!
      TN_names = ExtractTnsFromOrderInternal(this->ORDER_tokens);
      cytnx_error_msg(TN_names.size() != utensors.size(),
                      "[ERROR][Network][Contract--planning] order assigned but the [%d] tensors "
                      "appears in ORDER does not match the # input tensors [%d]\n",
                      TN_names.size(), utensors.size());
      for (int i = 0; i < this->names.size(); i++) {
        auto it = std::find(TN_names.begin(), TN_names.end(), this->names[i]);
        cytnx_error_msg(
          it == std::end(TN_names),
          "[ERROR][Network][Contract--planning] TN: <%s> defined but is not used in ORDER line\n",
          this->names[i].c_str());
        TN_names.erase(it);
      }
      if (TN_names.size() != 0) {
        std::cerr << "[ERROR] Following TNs appeared in ORDER line, but is not defined."
                  << std::endl;
        for (int i = 0; i < TN_names.size(); i++) {
          std::cerr << "        " << TN_names[i] << std::endl;
        }
        cytnx_error_msg(true, "%s", "\n");
      }

    }  // check all RN.

    // checking label matching:
    std::map<std::string, cytnx_int64> labelcnt;
    for (int i = 0; i < this->names.size(); i++) {
      for (int j = 0; j < this->label_arr[i].size(); j++) {
        if (labelcnt.find(this->label_arr[i][j]) == labelcnt.end())
          labelcnt[this->label_arr[i][j]] = 1;
        else
          labelcnt[this->label_arr[i][j]] += 1;
      }
    }
    std::vector<std::string> expected_TOUT;
    for (std::map<std::string, cytnx_int64>::iterator it = labelcnt.begin(); it != labelcnt.end();
         ++it) {
      if (it->second == 1) expected_TOUT.push_back(it->first);
    }
    bool err = false;
    if (expected_TOUT.size() != TOUT_labels.size()) {
      std::cout << expected_TOUT.size() << std::endl;
      err = true;
    }
    std::vector<std::string> itrsct = vec_intersect(expected_TOUT, this->TOUT_labels);
    if (itrsct.size() != expected_TOUT.size()) {
      err = true;
    }

    if (err) {
      std::cerr
        << "[ERROR][Network][Contract--planning] The TOUT contains labels that does not match "
           "with the delcartion from TNs.\n";
      std::cerr << "  > The reduced labels [rank:" << expected_TOUT.size() << "] should be:";
      for (int i = 0; i < expected_TOUT.size(); i++) std::cerr << expected_TOUT[i] << " ";
      std::cerr << std::endl;
      std::cerr << "  > The TOUT [rank" << TOUT_labels.size() << "] specified is:";
      for (int i = 0; i < TOUT_labels.size(); i++) std::cerr << TOUT_labels[i] << " ";
      std::cerr << std::endl;
      cytnx_error_msg(true, "%s", "\n");
    }

    // put tensor:
    for (int i = 0; i < utensors.size(); i++) this->tensors[i] = utensors[i];
  }

  void RegularNetwork::FromString(const std::vector<std::string> &contents) {
    this->clear();

    std::string line;
    std::vector<std::string> tmpvs;
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
        line.find_first_of(":") == std::string::npos,
        "[ERROR][Network][Fromfile] invalid network description at line: %d. should contain \':\'",
        i);

      tmpvs = str_split(line, false, ":");  // note that checking empty std::string!
      cytnx_error_msg(tmpvs.size() != 2,
                      "[ERROR][Network] invalid line in network file at line: %d. \n", i);

      std::string name = str_strip(tmpvs[0]);
      std::string content = str_strip(tmpvs[1]);

      // check if name contain invalid keyword or not assigned.
      cytnx_error_msg(name.length() == 0,
                      "[ERROR][Network][Fromfile] invalid tensor name at line: %d\n", i);
      cytnx_error_msg(name.find_first_of(" ;,") != std::string::npos,
                      "[ERROR] invalid Tensor name at line %d\n", i);

      // dispatch
      if (name == "ORDER") {
        if (content.length()) {
          // cut the line into tokens,
          // and leave it to process by CtTree after read all lines.
          this->order_line = content;
          this->ORDER_tokens = ParseOrderLineInternal(content, i);
          isORDER_exist = true;
        }
      } else if (name == "TOUT") {
        // if content has length, then pass to process.
        if (content.length()) {
          // this is an internal function that is defined in this cpp file.
          this->TOUT_labels = ParseToutLineInternal(this->TOUT_iBondNum, content, i);
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
        cytnx_error_msg(name.find_first_of("\t;\n: ") != std::string::npos,
                        "[ERROR][Network][Fromfile] invalid tensor name. cannot contain [' '] or "
                        "[';'] or [':'] or ['\\t'] .%s",
                        "\n");

        // check exists content:
        cytnx_error_msg(content.length() == 0,
                        "[ERROR][Network][Fromfile] invalid tensor labelsat line %d. cannot have "
                        "empty labels for input tensor. \n",
                        i);

        this->name2pos[name] = names.size() - 1;  // register
        this->label_arr.push_back(std::vector<std::string>());
        // this is an internal function that is defined in this cpp file.
        this->label_arr.back() = ParseTnLineInternal(content, i);
        this->iBondNums.push_back(this->label_arr.back().size());
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
    this->CtTree.base_nodes.clear();

    // Create base nodes properly
    for (std::size_t i = 0; i < this->names.size(); i++) {
      auto node = std::make_shared<Node>();
      node->name = this->names[i];
      this->CtTree.base_nodes.push_back(node);
    }

    // checking if all TN are set in ORDER.
    if (isORDER_exist) {
      std::vector<std::string> TN_names;
      TN_names = ExtractTnsFromOrderInternal(this->ORDER_tokens);
      for (int i = 0; i < this->names.size(); i++) {
        auto it = std::find(TN_names.begin(), TN_names.end(), this->names[i]);
        cytnx_error_msg(
          it == std::end(TN_names),
          "[ERROR][Network][Fromfile] TN: <%s> defined but is not used in ORDER line\n",
          this->names[i].c_str());
        TN_names.erase(it);
      }
      if (TN_names.size() != 0) {
        std::cerr << "[ERROR] Following TNs appeared in ORDER line, but is not defined."
                  << std::endl;
        for (int i = 0; i < TN_names.size(); i++) {
          std::cerr << "        " << TN_names[i] << std::endl;
        }
        cytnx_error_msg(true, "%s", "\n");
      }
    }  // check all RN.

    // checking label matching:
    std::map<std::string, cytnx_int64> labelcnt;
    for (int i = 0; i < this->names.size(); i++) {
      for (int j = 0; j < this->label_arr[i].size(); j++) {
        if (labelcnt.find(this->label_arr[i][j]) == labelcnt.end())
          labelcnt[this->label_arr[i][j]] = 1;
        else
          labelcnt[this->label_arr[i][j]] += 1;
      }
    }
    std::vector<std::string> expected_TOUT;
    for (std::map<std::string, cytnx_int64>::iterator it = labelcnt.begin(); it != labelcnt.end();
         ++it) {
      if (it->second == 1) expected_TOUT.push_back(it->first);
    }
    bool err = false;
    if (expected_TOUT.size() != TOUT_labels.size()) {
      err = true;
    }
    std::vector<std::string> itrsct = vec_intersect(expected_TOUT, this->TOUT_labels);
    if (itrsct.size() != expected_TOUT.size()) {
      err = true;
    }

    if (err) {
      std::cerr
        << "[ERROR][Network][Fromfile] The TOUT contains labels that does not match with the "
           "delcartion from TNs.\n";
      std::cerr << "  > The reduced labels [rank:" << expected_TOUT.size() << "] should be:";
      for (int i = 0; i < expected_TOUT.size(); i++) std::cerr << expected_TOUT[i] << " ";
      std::cerr << std::endl;
      std::cerr << "  > The TOUT [rank" << TOUT_labels.size() << "] specified is:";
      for (int i = 0; i < TOUT_labels.size(); i++) std::cerr << TOUT_labels[i] << " ";
      std::cerr << std::endl;
      cytnx_error_msg(true, "%s", "\n");
    }

    // maintain TOUT leg position
    TOUT_pos = std::vector<std::pair<int, int>>();
    for (int i = 0; i < TOUT_labels.size(); i++) {
      for (int j = 0; j < label_arr.size(); j++) {
        std::vector<std::string>::iterator it;
        it = std::find(this->label_arr[j].begin(), this->label_arr[j].end(), TOUT_labels[i]);
        if (it != this->label_arr[j].end()) {
          TOUT_pos.push_back(std::make_pair(j, distance(label_arr[j].begin(), it)));
          break;
        }
      }
    }

    // get int_label
    std::map<std::string, cytnx_int64> labelmap = std::map<std::string, cytnx_int64>();
    this->int_modes = std::vector<std::vector<cytnx_int64>>(this->label_arr.size());
    this->int_out_mode = std::vector<cytnx_int64>(this->TOUT_labels.size());
    cytnx_int64 label_int = 0;
    for (std::size_t i = 0; i < this->label_arr.size(); i++) {
      this->int_modes[i] = std::vector<cytnx_int64>(this->label_arr[i].size());
      for (std::size_t j = 0; j < this->label_arr[i].size(); j++) {
        labelmap.insert(std::pair<std::string, cytnx_int64>(this->label_arr[i][j], label_int));
        this->int_modes[i][j] = labelmap[this->label_arr[i][j]];
        label_int += 1;
      }
    }
    for (std::size_t i = 0; i < TOUT_labels.size(); i++) {
      this->int_out_mode[i] = labelmap[this->TOUT_labels[i]];
    }

  #ifdef UNI_GPU
    #ifdef UNI_CUQUANTUM
    this->optimizerInfo = nullptr;
    #endif
  #endif

    std::vector<std::string> names;
    for (int i = 0; i < this->names.size(); i++) {
      names.push_back(this->names[i]);
      CtTree.base_nodes[i]->name = this->names[i];
    }
    if (ORDER_tokens.size() != 0) {
      CtTree.build_contraction_tree_by_tokens(this->name2pos, ORDER_tokens);
    } else {
      CtTree.build_default_contraction_tree();
    }
    this->einsum_path = CtTreeToEinsumpathInternal(CtTree, names);
  }  // end of FromString

  void RegularNetwork::Fromfile(const std::string &fname) {
    const cytnx_uint64 MAXLINES = 1024;

    // empty all
    // this->clear();

    // open file
    std::ifstream infile;
    infile.open(fname.c_str());
    if (!(infile.is_open())) {
      cytnx_error_msg(true, "[Network] Error in opening file \'", fname.c_str(), "\'.\n");
    }
    filename = fname;

    std::string line;
    cytnx_uint64 lnum = 0;

    std::vector<std::string> contents;

    // read each line:
    while (lnum < MAXLINES) {
      lnum++;
      std::getline(infile, line);
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

  void RegularNetwork::PutUniTensors(const std::vector<std::string> &names,
                                     const std::vector<UniTensor> &utensors) {
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
  void RegularNetwork::RmUniTensor(const std::string &name) {
    cytnx_uint64 idx;
    try {
      idx = this->name2pos.at(name);
    } catch (std::out_of_range) {
      cytnx_error_msg(true,
                      "[ERROR][RegularNetwork][RmUniTensor] Cannot find the tensor name: [%s] in "
                      "current network.\n",
                      name.c_str());
    }

    this->RmUniTensor(idx);
  }
  void RegularNetwork::RmUniTensors(const std::vector<std::string> &names) {
    for (int i = 0; i < names.size(); i++) {
      this->RmUniTensor(names[i]);
    }
  }

  void RegularNetwork::Savefile(const std::string &fname) {
    cytnx_error_msg(
      this->label_arr.size() == 0,
      "[ERROR][RegularNetwork][Savefile] Cannot save empty network to network file!%s", "\n");

    std::fstream fo;
    fo.open(fname + ".net", std::ios::out | std::ios::trunc);
    if (!fo.is_open()) {
      cytnx_error_msg(true, "[ERROR][RegularNetwork][Savefile] Cannot open/create file:%s\n",
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
      fo << std::endl;
    }

    fo.close();
  }

  void RegularNetwork::PutUniTensor(const std::string &name, const UniTensor &utensor) {
    cytnx_uint64 idx;
    try {
      idx = this->name2pos.at(name);
    } catch (std::out_of_range) {
      cytnx_error_msg(true,
                      "[ERROR][RegularNetwork][PutUniTensor] Cannot find the tensor name: [%s] in "
                      "current network.\n",
                      name.c_str());
    }

    this->PutUniTensor(idx, utensor);
  }

  void RegularNetwork::PrintNet(std::ostream &os) {
    std::string status;
    os << "==== Network ====" << std::endl;
    if (this->tensors.size() == 0) {
      os << "      Empty      " << std::endl;
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
        os << std::endl;
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
      os << std::endl;
      os << "ORDER : ";
      for (cytnx_int64 i = 0; i < this->ORDER_tokens.size(); i++) {
        os << this->ORDER_tokens[i];
      }
      os << std::endl;
      os << "=================" << std::endl;
    }
  }

  std::string RegularNetwork::getOrder() {
    if (this->order_line != "") {
      return this->order_line;
    } else {
      return "No optimal or user specified order found.";
    }
  }

  void RegularNetwork::setOrder(const bool &optimal,
                                const std::string &contract_order /*default ""*/) {
    cytnx_warning_msg(optimal && (contract_order != ""),
                      "[WARNING][setOrder][RegularNetwork] Setting Optimal = true while specifying "
                      "the order, will find the optimal order instead."
                      "to use the desired order please set Optimal = false.%s",
                      "\n");
    cytnx_warning_msg(
      (!optimal) && (contract_order == ""),
      "[WARNING][setOrder][RegularNetwork] Setting Optimal = false while not "
      "specifying the order std::string, will use default contraciton order instrad.%s",
      "\n");
    this->ORDER_tokens.clear();
    if (optimal) {
      CheckInternal(this->tensors, this->names);

      if (this->tensors[0].device() == Device.cpu) {
        std::string Optim_ORDERline = this->getOptimalOrder();
        this->order_line = Optim_ORDERline;
        ORDER_tokens = ParseOrderLineInternal(Optim_ORDERline, 999999);
      } else {
  #ifdef UNI_GPU
    #ifdef UNI_CUQUANTUM
        if (this->tensors[0].uten_type() != UTenType.Dense) {
          std::string Optim_ORDERline = this->getOptimalOrder();
          this->order_line = Optim_ORDERline;
          ORDER_tokens = ParseOrderLineInternal(Optim_ORDERline, 999999);
        } else {
          std::vector<cytnx_uint64> out_shape;
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
          std::vector<std::pair<cytnx_int64, cytnx_int64>> path = cutn.getContractionPath();
          cutn.freeHandle();

          std::vector<std::string> names;
          for (int i = 0; i < this->names.size(); i++) {
            names.push_back(this->names[i]);
          }
          this->einsum_path = path;
          this->order_line = EinsumpathToStringInternal(path, names);
        }
    #else
        // cytnx_error_msg(true, "[ERROR][setOrder][RegularNetwork] fatal error,%s",
        //                 "try to call the gpu section for finding optimal contraction order
        //                 without " "CUQUANTUM support.\n");
        std::string Optim_ORDERline = this->getOptimalOrder();
        this->order_line = Optim_ORDERline;
        ORDER_tokens = ParseOrderLineInternal(Optim_ORDERline, 999999);
    #endif

  #else
        cytnx_error_msg(true, "[ERROR][setOrder][RegularNetwork] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
  #endif
      }

    } else {
      this->order_line = contract_order;
      if (contract_order != "") {
        ORDER_tokens = ParseOrderLineInternal(contract_order, 999999);
        CtTree.build_contraction_tree_by_tokens(this->name2pos, ORDER_tokens);
        this->einsum_path = CtTreeToEinsumpathInternal(CtTree, names);
      } else {
        CtTree.build_default_contraction_tree();
      }
    }
  }

  std::string RegularNetwork::getOptimalOrder() {
    // Creat a SearchTree to search for optim contraction order.
    SearchTree Stree;
    Stree.base_nodes.resize(this->tensors.size());
    for (cytnx_uint64 t = 0; t < this->tensors.size(); t++) {
      Stree.base_nodes[t].from_utensor(this->tensors[t]);  // create pseudotensors from base tensors
      // Stree.base_nodes[t].from_utensor(CtTree.base_nodes[t].utensor);
      Stree.base_nodes[t].accu_str = this->names[t];
    }
    Stree.search_order();
    return Stree.get_root().back()[0]->accu_str;
  }

  UniTensor RegularNetwork::Launch() {
    CheckInternal(this->tensors, this->names);

    int tn_device = this->tensors[0].device();

  #if defined(UNI_GPU) && defined(UNI_CUQUANTUM)  // gpu workflow with cuquantum
    if (tn_device != Device.cpu && this->tensors[0].uten_type() == UTenType.Dense) {
      std::vector<cytnx_uint64> out_shape;
      out_shape.reserve(this->TOUT_labels.size());
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
      if (this->optimizerInfo != nullptr) {
        cutn.setOptimizerInfo(this->optimizerInfo);
      } else {
        this->optimizerInfo = cutn.createOptimizerInfo();
      }
      cutn.setContractionPath(einsum_path);

      cutn.createWorkspaceDescriptor();
      cutn.initializePlan();
      cutn.autotune();
      cutn.executeContraction();
      cutn.freePlan();
      cutn.freeWorkspaceDescriptor();
      cutn.freeHandle();

      return out;
    }
  #endif

    // all other cases (CPU, without cuquantum, or not dense)
    for (cytnx_uint64 idx = 0; idx < this->tensors.size(); idx++) {
      this->CtTree.base_nodes[idx]->utensor =
        this->tensors[idx].relabel(this->label_arr[idx]);  // this conflict
      this->CtTree.base_nodes[idx]->is_assigned = true;
    }
    // 1.5 contraction order:
    if (ORDER_tokens.size() != 0) {
      // *set by user or optimally found
      CtTree.build_contraction_tree_by_tokens(this->name2pos, ORDER_tokens);
    } else {
      CtTree.build_default_contraction_tree();
    }

    // 2. contract using postorder traversal:
    std::stack<std::shared_ptr<Node>> stk;
    std::shared_ptr<Node> root = this->CtTree.nodes_container.back();
    root->set_root_ptrs();  // Add this line
    bool ict;

    do {
      // move the leftmost
      while (root != nullptr) {
        if (root->right) stk.push(root->right);
        stk.push(root);
        root = root->left;
      }

      root = stk.top();
      stk.pop();

      ict = true;
      if (root->right && !stk.empty()) {
        if (stk.top() == root->right) {  // This comparison now works with shared_ptr
          stk.pop();
          stk.push(root);
          root = root->right;
          ict = false;
        }
      }

      if (ict) {
        if (root->right && root->left) {
          root->utensor = Contract(root->left->utensor, root->right->utensor);
          root->left->clear_utensor();
          root->right->clear_utensor();
          root->is_assigned = true;
        }
        root = nullptr;
      }
    } while (!stk.empty());

    // 3. get result:
    UniTensor out = this->CtTree.nodes_container.back()->utensor;

    // 4. reset nodes:
    this->CtTree.reset_nodes();

    // 5. permute according to pre-set labels:
    if (TOUT_labels.size()) {
      out.permute_(TOUT_labels, TOUT_iBondNum);
    }
    return out;
  }

  void RegularNetwork::construct(const std::vector<std::string> &alias,
                                 const std::vector<std::vector<std::string>> &labels,
                                 const std::vector<std::string> &outlabel, const cytnx_int64 &outrk,
                                 const std::string &order, const bool optim) {
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
      this->ORDER_tokens = ParseOrderLineInternal(order, 0);
      std::vector<std::string> TN_names;
      TN_names = ExtractTnsFromOrderInternal(this->ORDER_tokens);
      for (int i = 0; i < this->names.size(); i++) {
        auto it = std::find(TN_names.begin(), TN_names.end(), this->names[i]);
        cytnx_error_msg(
          it == std::end(TN_names),
          "[ERROR][Network][Fromfile] TN: <%s> defined but is not used in ORDER line\n",
          this->names[i].c_str());
        TN_names.erase(it);
      }
      if (TN_names.size() != 0) {
        std::cerr << "[ERROR] Following TNs appeared in ORDER line, but is not defined."
                  << std::endl;
        for (int i = 0; i < TN_names.size(); i++) {
          std::cerr << "        " << TN_names[i] << std::endl;
        }
        cytnx_error_msg(true, "%s", "\n");
      }
    }  // check all RN.

    cytnx_error_msg(
      this->names.size() < 2,
      "[ERROR][Network][Fromfile] invalid network file. Should have at least 2 tensors defined.%s",
      "\n");

    this->tensors.resize(this->names.size());
    this->CtTree.base_nodes.clear();

    // Create nodes using make_shared
    for (std::size_t i = 0; i < this->names.size(); i++) {
      auto node = std::make_shared<Node>();
      node->name = this->names[i];
      this->CtTree.base_nodes.push_back(node);
    }

    // checking label matching:
    std::map<std::string, cytnx_int64> labelcnt;
    for (int i = 0; i < this->names.size(); i++) {
      for (int j = 0; j < this->label_arr[i].size(); j++) {
        if (labelcnt.find(this->label_arr[i][j]) == labelcnt.end())
          labelcnt[this->label_arr[i][j]] = 1;
        else
          labelcnt[this->label_arr[i][j]] += 1;
      }
    }
    std::vector<std::string> expected_TOUT;

    for (std::map<std::string, cytnx_int64>::iterator it = labelcnt.begin(); it != labelcnt.end();
         ++it) {
      if (it->second == 1) expected_TOUT.push_back(it->first);
    }
    if (this->TOUT_labels.size() == 0) {
      this->TOUT_labels = expected_TOUT;
    } else {
      bool err = false;
      if (expected_TOUT.size() != TOUT_labels.size()) {
        err = true;
      }
      std::vector<std::string> itrsct = vec_intersect(expected_TOUT, this->TOUT_labels);
      if (itrsct.size() != expected_TOUT.size()) {
        err = true;
      }
      if (err) {
        std::cerr
          << "[ERROR][Network][Fromfile] The TOUT contains labels that does not match with the "
             "delcartion from TNs.\n";
        std::cerr << "  > The reduced labels [rank:" << expected_TOUT.size() << "] should be:";
        for (int i = 0; i < expected_TOUT.size(); i++) std::cerr << expected_TOUT[i] << " ";
        std::cerr << std::endl;
        std::cerr << "  > The TOUT [rank" << TOUT_labels.size() << "] specified is:";
        for (int i = 0; i < TOUT_labels.size(); i++) std::cerr << TOUT_labels[i] << " ";
        std::cerr << std::endl;
        cytnx_error_msg(true, "%s", "\n");
      }
    }

    // maintain TOUT leg position
    TOUT_pos = std::vector<std::pair<int, int>>();
    for (int i = 0; i < TOUT_labels.size(); i++) {
      for (int j = 0; j < label_arr.size(); j++) {
        std::vector<std::string>::iterator it;
        it = std::find(this->label_arr[j].begin(), this->label_arr[j].end(), TOUT_labels[i]);
        if (it != this->label_arr[j].end()) {
          TOUT_pos.push_back(std::make_pair(j, distance(label_arr[j].begin(), it)));
          break;
        }
      }
    }

    // get int_label
    std::map<std::string, cytnx_int64> labelmap = std::map<std::string, cytnx_int64>();
    this->int_modes = std::vector<std::vector<cytnx_int64>>(this->label_arr.size());
    this->int_out_mode = std::vector<cytnx_int64>(this->TOUT_labels.size());
    cytnx_int64 label_int = 0;
    for (std::size_t i = 0; i < this->label_arr.size(); i++) {
      this->int_modes[i] = std::vector<cytnx_int64>(this->label_arr[i].size());
      for (std::size_t j = 0; j < this->label_arr[i].size(); j++) {
        labelmap.insert(std::pair<std::string, cytnx_int64>(this->label_arr[i][j], label_int));
        this->int_modes[i][j] = labelmap[this->label_arr[i][j]];
        label_int += 1;
      }
    }
    for (std::size_t i = 0; i < TOUT_labels.size(); i++) {
      this->int_out_mode[i] = labelmap[this->TOUT_labels[i]];
    }

  #ifdef UNI_GPU
    #ifdef UNI_CUQUANTUM
    this->optimizerInfo = nullptr;
    #endif
  #endif

    std::vector<std::string> names;
    for (int i = 0; i < this->names.size(); i++) {
      names.push_back(this->names[i]);
      CtTree.base_nodes[i]->name = this->names[i];
    }
    if (ORDER_tokens.size() != 0) {
      CtTree.build_contraction_tree_by_tokens(this->name2pos, ORDER_tokens);
    } else {
      CtTree.build_default_contraction_tree();
    }
    this->einsum_path = CtTreeToEinsumpathInternal(CtTree, names);
  }  // end construct

}  // namespace cytnx
#endif
