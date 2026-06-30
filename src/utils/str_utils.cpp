#include "utils/str_utils.hpp"
#include "Type.hpp"

namespace cytnx {

  std::vector<std::string> str_split(const std::string &in, const bool remove_null,
                                     const std::string &delimiter) {
    std::vector<std::string> out;
    std::size_t last = 0;
    std::size_t next = 0;
    std::string tmps;
    while ((next = in.find(delimiter, last)) != std::string::npos) {
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

  std::string str_strip(const std::string &in, const std::string &key) {
    if (in.empty()) return in;

    std::string tmp = in;  // make  copy of in string

    std::string::size_type pos = tmp.find_first_not_of(key);
    if (pos == std::string::npos) return std::string();
    tmp.erase(0, pos);  // ltrim

    pos = tmp.find_last_not_of(key);
    if (pos == std::string::npos) return tmp;
    tmp.erase(pos + 1);  // rtrim

    return tmp;
  }

  std::vector<std::string> str_findall(const std::string &in, const std::string &tokens) {
    std::vector<std::string> out;
    if (in.empty()) return out;

    std::size_t pos = 0, endpos;
    std::string tmp, op;
    if ((endpos = in.find_first_of(tokens, pos)) == std::string::npos) {
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

  std::string operator*(const std::string &in, const unsigned int &N) {
    std::string out = in;
    if (N == 0) return std::string("");
    for (cytnx_uint64 i = 1; i < N; i++) {
      out += in;
    }
    return out;
  }

}  // namespace cytnx
