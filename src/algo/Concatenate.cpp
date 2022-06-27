#include "algo.hpp"
#include "algo_internal_interface.hpp"
#include "Accessor.hpp"
#include "Generator.hpp"
namespace cytnx {
  namespace algo {
    typedef Accessor ac;
    Tensor Concatenate(Tensor T1, Tensor T2) {
      Tensor out;

      // check:
      cytnx_error_msg(T1.shape().size() != 1,
                      "[ERROR] concatenate currently can only accept 1d Tensor @T1.%s", "\n");
      cytnx_error_msg(T2.shape().size() != 1,
                      "[ERROR] concatenate currently can only accept 1d Tensor @T2.%s", "\n");
      cytnx_error_msg(T1.dtype() == Type.Void, "[ERROR] T1 is not initialize. dtype = Type.Void %s",
                      "\n");
      cytnx_error_msg(T2.dtype() == Type.Void, "[ERROR] T2 is not initialize. dtype = Type.Void %s",
                      "\n");

      cytnx_error_msg(T1.device() != T2.device(), "[ERROR] T1 and T2 should on same device.%s",
                      "\n");

      Tensor t1, t2;
      cytnx_int64 dtype;
      if (T1.dtype() < T2.dtype()) {
        t1 = T1;
        t2 = T2.astype(T1.dtype());
        dtype = t1.dtype();
      } else {
        t1 = T1.astype(T2.dtype());
        t2 = T2;
        dtype = t2.dtype();
      }

      out = zeros(t1.shape()[0] + t2.shape()[0], dtype, t1.device());

      out(ac::range(0, t1.shape()[0])) = t1;
      out(ac::tilend(t1.shape()[0])) = t2;

      return out;
    }
  }  // namespace algo
}  // namespace cytnx
