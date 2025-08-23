#ifndef CYTNX_TYPE_PRIORITY_H_
#define CYTNX_TYPE_PRIORITY_H_

#include "Type.hpp"

namespace cytnx {
  template <typename T>
  struct type_priority {
    static constexpr int value = 0;
  };

  template <>
  struct type_priority<cytnx_complex128> {
    static constexpr int value = 11;
  };
  template <>
  struct type_priority<cytnx_complex64> {
    static constexpr int value = 10;
  };
  template <>
  struct type_priority<cytnx_double> {
    static constexpr int value = 9;
  };
  template <>
  struct type_priority<cytnx_float> {
    static constexpr int value = 8;
  };
  template <>
  struct type_priority<cytnx_int64> {
    static constexpr int value = 7;
  };
  template <>
  struct type_priority<cytnx_uint64> {
    static constexpr int value = 6;
  };
  template <>
  struct type_priority<cytnx_int32> {
    static constexpr int value = 5;
  };
  template <>
  struct type_priority<cytnx_uint32> {
    static constexpr int value = 4;
  };
  template <>
  struct type_priority<cytnx_int16> {
    static constexpr int value = 3;
  };
  template <>
  struct type_priority<cytnx_uint16> {
    static constexpr int value = 2;
  };
  template <>
  struct type_priority<cytnx_bool> {
    static constexpr int value = 1;
  };
}  // namespace cytnx

#endif  // CYTNX_TYPE_PRIORITY_H_
