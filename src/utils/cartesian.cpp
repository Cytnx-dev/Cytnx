#include "utils/cartesian.hpp"
#include <algorithm>
#include <vector>
#include <cstring>
namespace cytnx {

  std::vector<cytnx_uint64> c2cartesian(const cytnx_uint64 &c,
                                        const std::vector<cytnx_uint64> &accu_shape) {
    cytnx_uint64 tmp = c;
    std::vector<cytnx_uint64> out(accu_shape.size());
    for (cytnx_uint64 i = 0; i < out.size(); i++) {
      out[i] = tmp / accu_shape[i];
      tmp = tmp % accu_shape[i];
    }
    return out;
  }

  cytnx_uint64 cartesian2c(const std::vector<cytnx_uint64> &vec,
                           const std::vector<cytnx_uint64> &accu_shape) {
    cytnx_uint64 out = 0;
    for (cytnx_uint64 i = 0; i < vec.size(); i++) out += vec[i] * accu_shape[i];

    return out;
  }
}  // namespace cytnx
