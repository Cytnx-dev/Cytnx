#include <typeinfo>
#include "Gncon.hpp"

#include "search_tree.hpp"
#include <stack>
#include <algorithm>
#include <iostream>

using namespace std;

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  // these three are internal functions:

  void _parse_task_line_(string line, vector<vector<pair<string, string>>> &table,
                         vector<std::string> &names, map<string, cytnx_uint64> &name2pos, int i) {
    vector<string> tmpvs, tmpvs_;
    tmpvs = str_split(line, false, "-");  // note that checking empty string!

    cytnx_error_msg(tmpvs.size() != 2, "[ERROR][Gncon] invalid line in Gncon file at line: %d. \n",
                    i);

    for (int j = 0; j < 2; j++) {
      tmpvs[j] = str_strip(tmpvs[j]);  // remove spaces

      // A valid line should contain ':'
      cytnx_error_msg(
        tmpvs[j].find_first_of(":") == string::npos,
        "[ERROR][Gncon][Fromfile] invalid Gncon description at line: %d. should contain \':\'", i);
      vector<string> task;
      tmpvs_ = str_split(tmpvs[j], false, ":");  // note that checking empty string!

      cytnx_error_msg(tmpvs_.size() != 2,
                      "[ERROR][Gncon] invalid line in Gncon file at line: %d. \n", i);

      tmpvs_[0] = str_strip(tmpvs_[0]);  // remove spaces
      tmpvs_[1] = str_strip(tmpvs_[1]);  // remove spaces

      // check if name contain invalid keyword or not assigned.
      cytnx_error_msg(tmpvs_[0].length() == 0,
                      "[ERROR][Gncon][Fromfile] invalid tensor name at line: %d\n", i);

      // check if name not exist:
      if (name2pos.find(tmpvs_[0]) == name2pos.end()) {
        names.push_back(tmpvs_[0]);
        table.push_back(vector<pair<string, string>>());
        // rep_labels.push_back(vector<pair<string, string>>());
        name2pos[tmpvs_[0]] = names.size() - 1;  // register
      }
      // std::cout<<"table size : "<<table.size()<<std::endl;
      // std::cout<<"index : "<<name2pos[tmpvs_[0]]<<std::endl;
      pair<string, string> ptmp(tmpvs_[1], "/" + to_string(i));
      // table[i] =  i-th tensor's leg names to be contracted, and its target labels.
      table[name2pos[tmpvs_[0]]].push_back(ptmp);
    }
  }

  void _parse_ORDER_line_(vector<string> &tokens, const string &line,
                          const cytnx_uint64 &line_num) {
    cytnx_error_msg((line.find_first_of("\t;\n:") != string::npos),
                    "[ERROR][Gncon][Fromfile] line:%d invalid ORDER line format.%s", line_num,
                    "\n");
    cytnx_error_msg((line.find_first_of("(),") == string::npos),
                    "[ERROR][Gncon][Fromfile] line:%d invalid ORDER line format.%s", line_num,
                    " tensors should be seperate by delimiter \',\' (comma), and/or wrapped with "
                    "\'(\' and \')\'");

    // check mismatch:
    size_t lbrac_n = std::count(line.begin(), line.end(), '(');
    size_t rbrac_n = std::count(line.begin(), line.end(), ')');
    cytnx_error_msg(lbrac_n != rbrac_n, "[ERROR][Gncon][Fromfile] parentheses mismatch.%s", "\n");

    // slice the line into pieces by parentheses and comma
    tokens = str_findall(line, "(),");

    cytnx_error_msg(tokens.size() == 0, "[ERROR][Gncon][Fromfile] line:%d invalid ORDER line.%s",
                    line_num, "\n");
  }
  void _parse_TOUT_line_(vector<cytnx_int64> &lbls, cytnx_uint64 &TOUT_iBondNum,
                         vector<vector<pair<string, string>>> &table,
                         map<string, cytnx_uint64> name2pos, const string &line,
                         const cytnx_uint64 &line_num) {
    // A:a->b,c->d ; B:b,c

    vector<string> tmp = str_split(line, false, ";");
    for (int i = 0; i < tmp.size(); i++) {
      tmp[i] = str_strip(tmp[i]);  // remove spaces
      vector<string> tmp_ = str_split(tmp[i], false, "/");
      string name = str_strip(tmp_[0]);
      string content = str_strip(tmp_[1]);
      // std::cout<<"name = "<<name<<std::endl;
      // std::cout<<"content = "<<content<<std::endl;
      vector<string> reps = str_split(content, false, ",");
      int tidx = name2pos[name];
      for (int j = 0; j < reps.size(); j++) {
        reps[j] = str_strip(reps[j]);  // remove spaces
        vector<string> ls = str_split(reps[j], false, ">");
        ls[0] = str_strip(ls[0]);  // remove spaces
        ls[1] = str_strip(ls[1]);  // remove spaces
        pair<string, string> tmp(ls[0], ls[1]);
        // std::cout<<"ls[0] = "<<ls[0]<<"ls[1] = "<<ls[1]<<std::endl;
        table[tidx].push_back(tmp);
      }
    }

    // // handle col-space lbl
    // vector<string> ket_lbls = str_split(tmp[0], false, ",");
    // if (ket_lbls.size() == 1)
    //   if (ket_lbls[0].length() == 0) ket_lbls.clear();
    // for (cytnx_uint64 i = 0; i < ket_lbls.size(); i++) {
    //   string tmp = str_strip(ket_lbls[i]);
    //   cytnx_error_msg(tmp.length() == 0,
    //                   "[ERROR][Gncon][Fromfile] line:%d Invalid labels for TOUT line.%s",
    //                   line_num, "\n");
    //   cytnx_error_msg((tmp.find_first_not_of("0123456789-") != string::npos),
    //                   "[ERROR][Gncon][Fromfile] line:%d %s\n", line_num,
    //                   "Invalid TOUT line. label contain non integer.");
    //   lbls.push_back(stoi(tmp, nullptr));
    // }
    // TOUT_iBondNum = lbls.size();

    // // handle row-space lbl
    // vector<string> bra_lbls = str_split(tmp[1], false, ",");
    // if (bra_lbls.size() == 1)
    //   if (bra_lbls[0].length() == 0) bra_lbls.clear();
    // for (cytnx_uint64 i = 0; i < bra_lbls.size(); i++) {
    //   string tmp = str_strip(bra_lbls[i]);
    //   cytnx_error_msg(tmp.length() == 0,
    //                   "[ERROR][Gncon][Fromfile] line:%d Invalid labels for TOUT line.%s",
    //                   line_num, "\n");
    //   cytnx_error_msg((tmp.find_first_not_of("0123456789-") != string::npos),
    //                   "[ERROR][Gncon][Fromfile] line:%d %s\n", line_num,
    //                   "Invalid TOUT line. label contain non integer.");
    //   lbls.push_back(stoi(tmp, nullptr));
    // }
  }

  /// This is debug function
  void print_gn(std::vector<vector<pair<string, string>>> &table, vector<string> &names,
                map<string, cytnx_uint64> &name2pos) {
    std::cout << "### table  ###" << std::endl;
    for (int i = 0; i < table.size(); i++) {
      for (int j = 0; j < table[i].size(); j++) {
        std::cout << "original lbl, replaced lbl =  " << table[i][j].first << " ,"
                  << table[i][j].second << std::endl;
      }
    }
    std::cout << "### tensor names  ###" << std::endl;
    for (int i = 0; i < names.size(); i++) {
      std::cout << names[i] << " pos : " << name2pos[names[i]] << std::endl;
    }
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

  void RegularGncon::Contract_plan(const std::vector<UniTensor> &utensors, const std::string &Tout,
                                   const std::vector<std::string> &alias,
                                   const std::string &contract_order) {
    // cytnx_error_msg(utensors.size() < 2,
    //                 "[ERROR][Gncon] invalid Gncon. Should have at least 2 tensors defined.%s",
    //                 "\n");

    // if (contract_order.length()) {
    //   // checing if alias is set!
    //   cytnx_error_msg(alias.size() == 0,
    //                   "[ERRPR] conraction_order need to be specify using alias name, so alias
    //                   name " "have to be assigned!%s",
    //                   "\n");
    // }

    // if (alias.size())
    //   cytnx_error_msg(utensors.size() != alias.size(),
    //                   "[ERROR] alias of UniTensor need to be assigned for all utensors.%s",
    //                   "\n");

    // bool isORDER_exist = false;
    // // reading
    // if (contract_order.length()) {
    //   // ORDER assigned
    //   _parse_ORDER_line_(this->ORDER_tokens, contract_order, 0);
    //   isORDER_exist = true;
    // }
    // if (Tout.length()) {
    //   // TOUT assigned
    //   _parse_TOUT_line_(this->TOUT_labels, this->TOUT_iBondNum, Tout, 0);
    // }

    // // assign input tensors into slots:
    // std::string name;
    // for (unsigned int i = 0; i < utensors.size(); i++) {
    //   if (alias.size()) {
    //     this->names.push_back(alias[i]);
    //     name = alias[i];
    //   } else {
    //     if (utensors[i].name().length()) {
    //       name = utensors[i].name() + "_usr";
    //     } else {
    //       name = "uname_T" + to_string(i);
    //     }
    //     this->names.push_back(name);
    //   }

    //   // check if name is valid:
    //   if (name2pos.find(name) != name2pos.end()) {
    //     cytnx_error_msg(true,
    //                     "[ERROR][Gncon][Fromfile] tensor name: [%s] has already exist. Cannot "
    //                     "have duplicated tensor name in a Gncon.",
    //                     name.c_str());
    //   }

    //   this->name2pos[name] = names.size() - 1;  // register
    //   // cout << name << "|" << names.size() - 1 << endl;
    //   this->label_arr.push_back(vector<cytnx_int64>());
    //   cytnx_uint64 tmp_iBN;
    //   // this is an internal function that is defined in this cpp file.
    //   this->label_arr.back() = utensors[i].labels();
    //   this->iBondNums.push_back(utensors[i].rowrank());
    //   //_parse_TN_line_(this->label_arr.back(),tmp_iBN,content,lnum);
    //   // this->iBondNums.push_back(tmp_iBN);

    // }  // traversal input tensor list

    // this->tensors.resize(this->names.size());
    // this->CtTree.base_nodes.resize(this->names.size());

    // // checking if all TN are set in ORDER.
    // //  only alias assigned will activate order
    // if (isORDER_exist) {
    //   std::vector<string> TN_names;  // this should be integer!
    //   _extract_TNs_from_ORDER_(TN_names, this->ORDER_tokens);
    //   cytnx_error_msg(TN_names.size() != utensors.size(),
    //                   "[ERROR][Gncon][Contract--planning] order assigned but the [%d] tensors "
    //                   "appears in ORDER does not match the # input tensors [%d]\n",
    //                   TN_names.size(), utensors.size());
    //   for (int i = 0; i < this->names.size(); i++) {
    //     auto it = std::find(TN_names.begin(), TN_names.end(), this->names[i]);
    //     cytnx_error_msg(
    //       it == std::end(TN_names),
    //       "[ERROR][Gncon][Contract--planning] TN: <%s> defined but is not used in ORDER line\n",
    //       this->names[i].c_str());
    //     TN_names.erase(it);
    //   }
    //   if (TN_names.size() != 0) {
    //     cout << "[ERROR] Following TNs appeared in ORDER line, but is not defined." << endl;
    //     for (int i = 0; i < TN_names.size(); i++) {
    //       cout << "        " << TN_names[i] << endl;
    //     }
    //     cytnx_error_msg(true, "%s", "\n");
    //   }

    // }  // check all RN.

    // // checking label matching:
    // map<cytnx_int64, cytnx_int64> lblcnt;
    // for (int i = 0; i < this->names.size(); i++) {
    //   for (int j = 0; j < this->label_arr[i].size(); j++) {
    //     if (lblcnt.find(this->label_arr[i][j]) == lblcnt.end())
    //       lblcnt[this->label_arr[i][j]] = 1;
    //     else
    //       lblcnt[this->label_arr[i][j]] += 1;
    //   }
    // }
    // vector<cytnx_int64> expected_TOUT;
    // for (map<cytnx_int64, cytnx_int64>::iterator it = lblcnt.begin(); it != lblcnt.end(); ++it) {
    //   if (it->second == 1) expected_TOUT.push_back(it->first);
    // }
    // bool err = false;
    // if (expected_TOUT.size() != TOUT_labels.size()) {
    //   std::cout << expected_TOUT.size() << std::endl;
    //   err = true;
    // }
    // vector<cytnx_int64> itrsct = vec_intersect(expected_TOUT, this->TOUT_labels);
    // if (itrsct.size() != expected_TOUT.size()) {
    //   err = true;
    // }

    // if (err) {
    //   cout << "[ERROR][Gncon][Contract--planning] The TOUT contains labels that does not match "
    //           "with the delcartion from TNs.\n";
    //   cout << "  > The reduced labels [rank:" << expected_TOUT.size() << "] should be:";
    //   for (int i = 0; i < expected_TOUT.size(); i++) cout << expected_TOUT[i] << " ";
    //   cout << endl;
    //   cout << "  > The TOUT [rank" << TOUT_labels.size() << "] specified is:";
    //   for (int i = 0; i < TOUT_labels.size(); i++) cout << TOUT_labels[i] << " ";
    //   cout << endl;
    //   cytnx_error_msg(true, "%s", "\n");
    // }

    // // put tensor:
    // for (int i = 0; i < utensors.size(); i++) this->tensors[i] = utensors[i];
  }

  void RegularGncon::FromString(const std::vector<std::string> &contents) {
    // "A:a-B:b","B:d-A:k"

    this->clear();

    string line;
    vector<string> tmpvs;

    bool isORDER_exist = false;

    for (int i = 0; i < contents.size(); i++) {
      vector<string> task;
      line = contents[i];
      line = str_strip(line, "\n");  // remove leading and ending /n
      line = str_strip(line);  // remove leading and ending spaces and tab
      if (line.length() == 0) continue;  // blank line
      if (line.at(0) == '#') continue;  // comment whole line.

      // remove any comment at eol :
      line = str_split(line, true, "#")[0];  // remove comment on end.
      // std::cout<<line<<std::endl;
      if (line.find_first_of("-") == string::npos) {
        // Not a contraction line, should be either ORDER or TOUT

        tmpvs = str_split(line, false, ":");  // note that checking empty string!
        string command = str_strip(tmpvs[0]);
        string content = str_strip(tmpvs[1]);
        // cytnx_error_msg(command.find_first_of(" ;,") != string::npos,
        //                 "[ERROR] invalid Tensor name at line %d\n", i);

        if (command == "ORDER") {
          if (content.length()) {
            // cut the line into tokens,
            // and leave it to process by CtTree after read all lines.
            _parse_ORDER_line_(this->ORDER_tokens, content, i);
            isORDER_exist = true;
          }
        } else if (command == "TOUT") {
          // if content has length, then pass to process.
          if (content.length()) {
            // this is an internal function that is defined in this cpp file.
            _parse_TOUT_line_(this->TOUT_labels, this->TOUT_iBondNum, this->table, this->name2pos,
                              content, i);
          }

        } else {
          cytnx_error_msg(true, "[ERROR][Gncon] invalid line in Gncon file at line: %d. \n", i);
        }

      } else {
        // Is a cotraction task line!
        _parse_task_line_(line, this->table, this->names, this->name2pos, i);
      }  // end line.find_first_of("-") == string::npos

    }  // end for i loop

    // cytnx_error_msg(lnum>=MAXLINES,"[ERROR][Gncon][Fromfile] Gncon file exceed the maxinum
    // allowed lines, MAXLINES=1024%s","\n");

    cytnx_error_msg(
      this->names.size() < 2,
      "[ERROR][Gncon][Fromfile] invalid Gncon file. Should have at least 2 tensors defined.%s",
      "\n");

    this->label_arr.resize(this->names.size());
    this->tensors.resize(this->names.size());
    this->CtTree.base_nodes.resize(this->names.size());

    // checking if all TN are set in ORDER.
    if (isORDER_exist) {
      std::vector<string> TN_names;
      _extract_TNs_from_ORDER_(TN_names, this->ORDER_tokens);
      for (int i = 0; i < this->names.size(); i++) {
        auto it = std::find(TN_names.begin(), TN_names.end(), this->names[i]);
        cytnx_error_msg(it == std::end(TN_names),
                        "[ERROR][Gncon][Fromfile] TN: <%s> defined but is not used in ORDER line\n",
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
    // print_gn(this->table, this->names, this->name2pos);
  }

  void RegularGncon::Fromfile(const std::string &fname) {
    const cytnx_uint64 MAXLINES = 1024;

    // empty all
    // this->clear();

    // open file
    std::ifstream infile;
    infile.open(fname.c_str());
    if (!(infile.is_open())) {
      cytnx_error_msg(true, "[Gncon] Error in opening file \'", fname.c_str(), "\'.\n");
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
      "[ERROR][Gncon][Fromfile] Gncon file exceed the maxinum allowed lines, MAXLINES=1024%s",
      "\n");

    this->FromString(contents);
  }

  void RegularGncon::PutUniTensors(const std::vector<string> &names,
                                   const std::vector<UniTensor> &utensors) {
    cytnx_error_msg(names.size() != utensors.size(),
                    "[ERROR][RegularGncon][PutUniTensors] total number of names does not match "
                    "number of input UniTensors.%s",
                    "\n");
    for (int i = 0; i < names.size(); i++) {
      this->PutUniTensor(names[i], utensors[i]);
    }
  }

  void RegularGncon::PutUniTensor(const cytnx_uint64 &idx, const UniTensor &utensor) {
    cytnx_error_msg(idx >= this->CtTree.base_nodes.size(),
                    "[ERROR][RegularGncon][PutUniTensor] index=%d out of range.\n", idx);

    // check shape:
    // cytnx_error_msg(this->label_arr[idx].size() != utensor.rank(),
    //                 "[ERROR][RegularGncon][PutUniTensor] tensor name: [%s], the rank of input "
    //                 "UniTensor does not match the definition in Gncon file.\n",
    //                 this->names[idx].c_str());
    // cytnx_error_msg(this->iBondNums[idx] != utensor.rowrank(),
    //                 "[ERROR][RegularGncon][PutUniTensor] tensor name: [%s], the row-rank of "
    //                 "input UniTensor does not match the semicolon defined in Gncon file.\n",
    //                 this->names[idx].c_str());

    // put tensor
    this->tensors[idx] = utensor;

    // update label_arr: clear
    this->label_arr[idx].resize(utensor.rank());
    // this->is_contracted[idx].resize(utensor.rank());

    // update label_arr: update
    for (int i = 0; i < this->table[idx].size(); i++) {
      pair<string, string> tmp = this->table[idx][i];
      int utidx = utensor.get_index(tmp.first);
      this->label_arr[idx][utidx] = tmp.second;
      // this->is_contracted[idx][utidx] = 1;
    }
  }

  void RegularGncon::Savefile(const std::string &fname) {
    cytnx_error_msg(this->label_arr.size() == 0,
                    "[ERROR][RegularGncon][Savefile] cannot save empty Gncon to Gncon file!%s",
                    "\n");

    fstream fo;
    fo.open(fname + ".net", ios::out | ios::trunc);
    if (!fo.is_open()) {
      cytnx_error_msg(true, "[ERROR][RegularGncon][Savefile] cannot open/create file:%s\n",
                      fname.c_str());
    }

    for (int i = 0; i < this->label_arr.size(); i++) {
      fo << this->names[i] << " : ";
      if (this->iBondNums[i] == 0) fo << ";";

      for (int j = 0; j < this->label_arr[i].size(); j++) {
        fo << this->label_arr[i][j];

        if (j + 1 == this->iBondNums[i])
          fo << ";";
        else if (j != this->label_arr[i].size() - 1)
          fo << ",";

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

  void RegularGncon::PutUniTensor(const std::string &name, const UniTensor &utensor) {
    cytnx_uint64 idx;
    /*
    std::cout << "|" << name <<"|" << std::endl;
    for(auto it=this->name2pos.begin();it!=this->name2pos.end();it++){
        std::cout << "|"<<it->first<<"|"<<it->second<<"|" << std::endl;
    }
    */
    try {
      idx = this->name2pos.at(name);
    } catch (std::out_of_range) {
      cytnx_error_msg(true,
                      "[ERROR][RegularGncon][PutUniTensor] cannot find the tensor name: [%s] in "
                      "current Gncon.\n",
                      name.c_str());
    }

    this->PutUniTensor(idx, utensor);
  }

  void RegularGncon::PrintNet(std::ostream &os) {
    string status;
    os << "==== Gncon ====" << endl;
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
        os << "; ";
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

  string RegularGncon::getOptimalOrder() {
    // Creat a SearchTree to search for optim contraction order.
    SearchTree Stree;
    Stree.base_nodes.resize(this->tensors.size());
    for (cytnx_uint64 t = 0; t < this->tensors.size(); t++) {
      // Stree.base_nodes[t].from_utensor(this->tensors[t]); //create psudotensors from base tensors
      Stree.base_nodes[t].from_utensor(CtTree.base_nodes[t].utensor);
      Stree.base_nodes[t].accu_str = this->names[t];
    }
    Stree.search_order();
    return Stree.nodes_container.back()[0].accu_str;
  }

  UniTensor RegularGncon::Launch(const bool &optimal, const string &contract_order /*default ""*/) {
    // 1. check tensors are all set, and put all unitensor on node for contraction:
    cytnx_error_msg(this->tensors.size() == 0,
                    "[ERROR][Launch][RegularGncon] cannot launch an un-initialize Gncon.%s", "\n");
    cytnx_error_msg(this->tensors.size() < 2,
                    "[ERROR][Launch][RegularGncon] Gncon should contain >=2 tensors.%s", "\n");

    // check not both optimal=true and contract_order not nullptr
    cytnx_error_msg(
      optimal and contract_order != "",
      "[ERROR][Launch][RegularGncon] cannot launch with optimal=True and given contract_order.%s",
      "\n");
    int outleg_lbl = -1;

    for (cytnx_uint64 idx = 0; idx < this->tensors.size(); idx++) {
      cytnx_error_msg(this->tensors[idx].uten_type() == UTenType.Void,
                      "[ERROR][Launch][RegularGncon] tensor at [%d], name: [%s] is not set.\n", idx,
                      this->names[idx].c_str());
      // transion save old labels:
      //  old_labels.push_back(this->tensors[idx].labels());

      // std::cout<<"idx: "<<idx<<" label: ";
      // for (cytnx_uint64 j = 0; j < this->label_arr[idx].size(); j++){
      //   std::cout<<this->label_arr[idx][j]<<",";
      // }
      // std::cout<<std::endl;

      // modify the label of unitensor (shared):
      //  this->tensors[idx].set_labels(this->label_arr[idx]);//this conflict
      this->CtTree.base_nodes[idx].utensor =
        this->tensors[idx].relabels(this->label_arr[idx]);  // this conflict
      // this->CtTree.base_nodes[idx].name = this->tensors[idx].name();
      this->CtTree.base_nodes[idx].is_assigned = true;

      // cout << this->tensors[idx].name() << " " << idx << "from dict:" <<
      // this->name2pos[this->tensors[idx].name()] << endl;
    }

    // 1.5 contraction order:
    if (ORDER_tokens.size() != 0) {
      // *set by user
      CtTree.build_contraction_tree_by_tokens(this->name2pos, ORDER_tokens);

    } else {
      if (optimal == true) {
        string Optim_ORDERline = this->getOptimalOrder();
        this->ORDER_tokens.clear();
        _parse_ORDER_line_(ORDER_tokens, Optim_ORDERline, 999999);
        CtTree.build_contraction_tree_by_tokens(this->name2pos, ORDER_tokens);
      } else if (contract_order != "") {
        this->ORDER_tokens.clear();
        _parse_ORDER_line_(ORDER_tokens, contract_order, 999999);
        CtTree.build_contraction_tree_by_tokens(this->name2pos, ORDER_tokens);
      } else {
        CtTree.build_default_contraction_tree();
      }
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
          // cout << "Contract:" << root->left->utensor.name() << " " << root->right->utensor.name()
          // << endl; root->left->utensor.print_diagram(); root->right->utensor.print_diagram();
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
    // std::cout << out << std::endl;
    // out.print_diagram();

    // 4. reset nodes:
    this->CtTree.reset_nodes();

    // //5. reset back the original labels:
    // for(cytnx_uint64 i=0;i<this->tensors.size();i++){
    //     this->tensors[i].set_labels(old_labels[i]);
    // }

    // 6. permute accroding to pre-set labels:
    if (TOUT_labels.size()) {
      out.permute_(TOUT_labels, TOUT_iBondNum, true);
    }

    // UniTensor out;
    return out;
  }

}  // namespace cytnx

#endif
