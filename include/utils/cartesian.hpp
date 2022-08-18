#ifndef __H_cartesian_
#define __H_cartesian_

#include <vector>
#include "Type.hpp"
namespace cytnx {

  cytnx_uint64 cartesian2c(const std::vector<cytnx_uint64> &vec,
                           const std::vector<cytnx_uint64> &accu_shape);
  std::vector<cytnx_uint64> c2cartesian(const cytnx_uint64 &c,
                                        const std::vector<cytnx_uint64> &accu_shape);

}  // namespace cytnx
#endif
