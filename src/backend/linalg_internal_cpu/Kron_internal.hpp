#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_KRON_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_KRON_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    template <class TO, class TL, class TR>
    void Kron_general(TO* out, const TL* Lin, const TR* Rin,
                      const std::vector<cytnx_uint64>& shape1,
                      const std::vector<cytnx_uint64>& shape2) {
      cytnx_error_msg(shape1.size() != shape2.size(),
                      "[ERROR][Internal Kron] T1 rank != T2 rank %s", "\n");
      cytnx_uint64 TotalElem = shape1[0] * shape2[0];
      std::vector<cytnx_uint64> new_shape_acc(shape1.size());
      std::vector<cytnx_uint64> shape1_acc(shape1.size());
      std::vector<cytnx_uint64> shape2_acc(shape2.size());
      new_shape_acc.back() = 1;
      shape1_acc.back() = 1;
      shape2_acc.back() = 1;

      for (unsigned long long i = 1; i < new_shape_acc.size(); i++) {
        new_shape_acc[new_shape_acc.size() - 1 - i] = new_shape_acc[new_shape_acc.size() - i] *
                                                      shape1[new_shape_acc.size() - i] *
                                                      shape2[new_shape_acc.size() - i];
        TotalElem *= shape1[i] * shape2[i];
        shape1_acc[shape1_acc.size() - 1 - i] =
          shape1_acc[shape1_acc.size() - i] * shape1[shape1_acc.size() - i];
        shape2_acc[shape2_acc.size() - 1 - i] =
          shape2_acc[shape2_acc.size() - i] * shape2[shape2_acc.size() - i];
      }

      for (unsigned long long i = 0; i < TotalElem; i++) {
        cytnx_uint64 tmp = i, tmp2;
        cytnx_uint64 x = 0, y = 0;
        for (unsigned long long j = 0; j < new_shape_acc.size(); j++) {
          tmp2 = tmp / new_shape_acc[j];
          tmp %= new_shape_acc[j];
          x += cytnx_uint64(tmp2 / shape2[j]) * shape1_acc[j];
          y += cytnx_uint64(tmp2 % shape2[j]) * shape2_acc[j];
        }
        out[i] = Lin[x] * Rin[y];
      }
    }

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_KRON_INTERNAL_H_
