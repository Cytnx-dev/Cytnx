#ifndef __utils_H_
#define __utils_H_

#include "cartesian.hpp"
#include "vec_cast.hpp"
#include "vec_clone.hpp"
#include "vec_unique.hpp"
#include "vec_map.hpp"
#include "vec_erase.hpp"
#include "vec_where.hpp"
#include "vec_concatenate.hpp"
#include "vec_intersect.hpp"
#include "vec_range.hpp"
#include "vec2d_col_sort.hpp"
#include "str_utils.hpp"
#include "vec_print.hpp"
#include "vec_io.hpp"
#include "vec_argsort.hpp"
#include "vec_sort.hpp"
#include "dynamic_arg_resolver.hpp"

#include "complex_arithmetic.hpp"
#include "cucomplex_arithmetic.hpp"

/// Helper function to print vector with ODT:
#include <vector>
#include <iostream>
#include <string>

namespace cytnx {
  bool _fx_compare_vec_inc(const std::vector<cytnx_int64> &v1, const std::vector<cytnx_int64> &v2);
  bool _fx_compare_vec_dec(const std::vector<cytnx_int64> &v1, const std::vector<cytnx_int64> &v2);
};  // namespace cytnx

#endif
