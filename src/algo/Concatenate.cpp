#include "algo.hpp"
#include "Accessor.hpp"
#include "Generator.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/algo_internal_cpu/Concate_internal.hpp"
  #ifdef UNI_GPU
    #include "backend/algo_internal_gpu/cuConcate_internal.hpp"
  #endif
  #include "backend/Storage.hpp"
  #include "backend/Scalar.hpp"
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

      // promote across the real/complex boundary (e.g. ComplexFloat + Double -> ComplexDouble)
      // rather than keeping the lower-enum operand type.
      cytnx_int64 dtype = Type.type_promote(T1.dtype(), T2.dtype());
      Tensor t1 = T1.astype(dtype);
      Tensor t2 = T2.astype(dtype);

      out = zeros({t1.shape()[0] + t2.shape()[0]}, dtype, t1.device());

      out(ac::range(0, t1.shape()[0])) = t1;
      out(ac::tilend(t1.shape()[0])) = t2;

      return out;
    }
  }  // namespace algo
}  // namespace cytnx

#endif  // BACKEND_TORCH
