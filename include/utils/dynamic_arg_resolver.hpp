#ifndef __H_dynamic_arg_resolver_
#define __H_dynamic_arg_resolver_

#include <vector>
#include "Type.hpp"
namespace cytnx {

  // elements resolver
  void _resolve_int64(std::vector<cytnx_int64> &cool);

  template <class T>
  void _resolve_int64(std::vector<cytnx_int64> &cool, const T &a) {
    cool.push_back(a);
  }

  template <class... Ts>
  void _resolve_int64(std::vector<cytnx_int64> &cool, const cytnx_int64 &a, const Ts &...args) {
    cool.push_back(a);
    _resolve_int64(cool, args...);
  }

  template <class... Ts>
  std::vector<cytnx_int64> dynamic_arg_int64_resolver(const cytnx_int64 &a, const Ts &...args) {
    // std::cout << a << std::endl;
    std::vector<cytnx_int64> idxs;
    _resolve_int64(idxs, a, args...);
    // cout << idxs << endl;
    return idxs;
  }

  //-----------------
  void _resolve_uint64(std::vector<cytnx_uint64> &cool);

  template <class T>
  void _resolve_uint64(std::vector<cytnx_uint64> &cool, const T &a) {
    cool.push_back(a);
  }

  template <class... Ts>
  void _resolve_uint64(std::vector<cytnx_uint64> &cool, const cytnx_uint64 &a, const Ts &...args) {
    cool.push_back(a);
    _resolve_uint64(cool, args...);
  }

  template <class... Ts>
  std::vector<cytnx_uint64> dynamic_arg_uint64_resolver(const cytnx_uint64 &a, const Ts &...args) {
    // std::cout << a << std::endl;;
    std::vector<cytnx_uint64> idxs;
    _resolve_uint64(idxs, a, args...);
    // cout << idxs << endl;
    return idxs;
  }

}  // namespace cytnx
#endif
