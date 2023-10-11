#include "cytnx.hpp"
#include "linalg.hpp"
#include "utils/utils.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
#include <numeric>
using namespace std;

#ifdef BACKEND_TORCH
#else

  #ifdef UNI_OMP
    #include <omp.h>
  #endif

namespace cytnx {
  namespace linalg {
    cytnx::Tensor Directsum(const cytnx::Tensor &T1, const cytnx::Tensor &T2,
                            const std::vector<cytnx_uint64> &shared_axes) {
      // check:
      cytnx_error_msg(T1.shape().size() != T2.shape().size(),
                      "[ERROR] T1 and T2 must be the same rank!%s", "\n");
      cytnx_error_msg(shared_axes.size() > T1.shape().size(),
                      "[ERROR] len(shared_axes) must be small or equal to the rank of Tensors.%s",
                      "\n");
      cytnx_error_msg(T1.device() != T2.device(), "[ERROR] Two tensors must be on same devices.%s",
                      "\n");

      // checking dulipcation in shared_axes:
      auto tmp = vec_unique(shared_axes);
      cytnx_error_msg(tmp.size() != shared_axes.size(),
                      "[ERROR] shared_axes cannot contain duplicate elements!%s", "\n");

      std::vector<cytnx_uint64> new_shape(T1.rank());
      // checking dimension in shared_axes:
      for (int i = 0; i < shared_axes.size(); i++) {
        cytnx_error_msg(shared_axes[i] >= T1.shape().size(),
                        "[ERROR] axis %d specify in shared_axes[%d] is out of bound!\n",
                        shared_axes[i], i);
        cytnx_error_msg(
          T1.shape()[shared_axes[i]] != T2.shape()[shared_axes[i]],
          "[ERROR] T1 and T2 at axis %d which specified to share does not have same dimension!\n",
          shared_axes[i]);
        new_shape[shared_axes[i]] = T1.shape()[shared_axes[i]];
      }

      std::vector<cytnx_uint64> non_shared_axes;
      for (int i = 0; i < new_shape.size(); i++) {
        if (new_shape[i] == 0) {
          new_shape[i] = T1.shape()[i] + T2.shape()[i];
          non_shared_axes.push_back(i);
        }
      }

      Tensor _t1 = T1.contiguous(), _t2 = T2.contiguous();
      if (T1.dtype() != T2.dtype()) {
        // do conversion:
        if (T1.dtype() < T2.dtype()) {
          _t2 = _t2.astype(T1.dtype());
        } else {
          _t1 = _t1.astype(T2.dtype());
        }
      }

      Tensor out(new_shape, _t1.dtype(), _t1.device());

      std::vector<Accessor> accs(out.rank());
      for (int i = 0; i < shared_axes.size(); i++) {
        accs[shared_axes[i]] = Accessor::all();
      }
      for (int i = 0; i < non_shared_axes.size(); i++) {
        accs[non_shared_axes[i]] = Accessor::range(0, T1.shape()[non_shared_axes[i]]);
      }
      out[accs] = _t1;

      for (int i = 0; i < non_shared_axes.size(); i++) {
        accs[non_shared_axes[i]] = Accessor::tilend(T1.shape()[non_shared_axes[i]]);
      }
      out[accs] = _t2;

      return out;
    };

  }  // namespace linalg
}  // namespace cytnx
#endif  // BACKEND_TORCH
