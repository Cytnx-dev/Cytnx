#ifndef _H_str_utils_
#define _H_str_utils_

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

namespace cytnx {

  // std::vector<std::string> str_split(const std::string &in, const std::vector<std::string>
  // &delimiters={" "}); std::string str_strip(const std::string &in,const std::vector<std::string>
  // &keys={" ","\n"});

  std::string str_strip(const std::string &in, const std::string &key = " \n\r");
  std::vector<std::string> str_split(const std::string &in, const bool remove_null = true,
                                     const std::string &delimiter = " ");
  std::vector<std::string> str_findall(const std::string &in, const std::string &tokens);

}  // namespace cytnx

#endif
