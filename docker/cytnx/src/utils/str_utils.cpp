#include "utils/str_utils.hpp"
#include "Type.hpp"
using namespace std;

namespace cytnx {

  vector<string> str_split(const string &in, const bool remove_null, const string &delimiter) {
    vector<string> out;
    size_t last = 0;
    size_t next = 0;
    string tmps;
    while ((next = in.find(delimiter, last)) != string::npos) {
      tmps = in.substr(last, next - last);
      if (remove_null) {
        if ((tmps != delimiter) && (tmps.length() != 0)) {
          out.push_back(tmps);
        }
      } else {
        out.push_back(tmps);
      }
      last = next + 1;
    }
    tmps = in.substr(last);
    if (remove_null) {
      if ((tmps != delimiter) && (tmps.length() != 0)) {
        out.push_back(tmps);
      }
    } else {
      out.push_back(tmps);
    }
    return out;
  }

  string str_strip(const string &in, const string &key) {
    if (in.empty()) return in;

    string tmp = in;  // make  copy of in string

    string::size_type pos = tmp.find_first_not_of(key);
    if (pos == string::npos) return string();
    tmp.erase(0, pos);  // ltrim

    pos = tmp.find_last_not_of(key);
    if (pos == string::npos) return tmp;
    tmp.erase(pos + 1);  // rtrim

    return tmp;
  }

  vector<string> str_findall(const string &in, const string &tokens) {
    vector<string> out;
    if (in.empty()) return out;

    size_t pos = 0, endpos;
    string tmp, op;
    if ((endpos = in.find_first_of(tokens, pos)) == string::npos) {
      out.push_back(in);
      return out;
    }
    tmp = in.substr(pos, endpos - pos + 1);
    op = tmp.back();
    tmp.pop_back();
    if (tmp.length()) out.push_back(tmp);
    out.push_back(op);
    pos = endpos + 1;

    while (((endpos = in.find_first_of(tokens, pos)) != std::string::npos)) {
      tmp = in.substr(pos, endpos - pos + 1);
      op = tmp.back();
      tmp.pop_back();
      if (tmp.length()) out.push_back(tmp);
      out.push_back(op);
      pos = endpos + 1;
    }
    tmp = in.substr(pos, in.length());
    if (tmp.length()) out.push_back(tmp);
    return out;
  }

}  // namespace cytnx
