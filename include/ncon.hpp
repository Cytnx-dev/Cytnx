#ifndef _ncon_H_
#define _ncon_H_

#include "cytnx.hpp"
#include <string>
#include <stack>
#include <vector>

namespace cytnx {
  UniTensor ncon(const std::vector<UniTensor> &tensor_list_in,
                 const std::vector<std::vector<cytnx_int64>> &connect_list_in,
                 const bool check_network = false, const bool optimize = false,
                 std::vector<cytnx_int64> cont_order = std::vector<cytnx_int64>(),
                 const std::vector<cytnx_int64> &out_labels = std::vector<cytnx_int64>());
}  // namespace cytnx

#endif
