#include "cytnx.hpp"
#include "linalg.hpp"
#include "utils/utils.hpp"
#include "Tensor.hpp"
#include <numeric>

#ifdef BACKEND_TORCH
#else

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
      cytnx_error_msg(T1.dtype() == Type.Void || T2.dtype() == Type.Void,
                      "[ERROR] input tensors cannot have dtype Type.Void.%s", "\n");
      cytnx_error_msg(T1.is_scalar() || T2.is_scalar(),
                      "[ERROR] Directsum does not support rank-0 scalar tensors.%s", "\n");

      // checking duplication in shared_axes:
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

      // promote across the real/complex boundary (e.g. ComplexFloat + Double -> ComplexDouble)
      // rather than keeping the lower-enum operand type; astype is a no-op when it already matches.
      const unsigned int out_dtype = Type.type_promote(T1.dtype(), T2.dtype());
      Tensor _t1 = T1.contiguous().astype(out_dtype);
      Tensor _t2 = T2.contiguous().astype(out_dtype);

      Tensor out(new_shape, out_dtype, _t1.device());

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
