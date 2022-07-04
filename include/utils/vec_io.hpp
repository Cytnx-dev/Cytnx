#ifndef __H_vec_io_
#define __H_vec_io_

#include <vector>
#include <cstring>
#include <fstream>
#include <typeinfo>
#include "cytnx_error.hpp"

namespace cytnx {

  // get filesize
  unsigned long long FileSize(const char *sFileName);

  //
  template <class T>
  void vec_tofile(std::fstream &f, const std::vector<T> &in) {
    cytnx_error_msg(!f.is_open(), "[ERROR][vec_tofile] fstream is not opened!%s", "\n");
    f.write((char *)&in[0], sizeof(T) * in.size());
  }

  template <class T>
  void vec_tofile(const std::string &filepath, const std::vector<T> &in) {
    std::fstream f(filepath, std::ios::out | std::ios::binary);
    cytnx_error_msg(!f.is_open(), "[ERROR][vec_tofile] cannot open filepath: %s\n", filepath);
    vec_tofile(f, in);
    f.close();
  }

  //----------------------------------------------------------------------------------------
  template <class T>
  std::vector<T> vec_fromfile(const std::string &filepath) {
    std::fstream f(filepath, std::ios::in | std::ios::binary);
    cytnx_error_msg(!f.is_open(), "[ERROR][vec_fromfile] cannot open filepath: %s\n", filepath);
    auto fsize = FileSize(filepath.c_str());
    cytnx_error_msg(fsize % sizeof(T),
                    "[ERROR][vec_fromfile] the total size of file %lu is not multiple of size %lu "
                    "with type: %s\n",
                    fsize, sizeof(T), typeid(T).name());

    std::vector<T> out(static_cast<unsigned long long>(fsize / sizeof(T)));
    f.read((char *)&out[0], sizeof(T) * out.size());

    f.close();
    return out;
  }

};  // namespace cytnx
#endif
