#include "cuMod_internal.hpp"

#include "cuArithmeticDispatch.cuh"

namespace cytnx {

  namespace linalg_internal {

    // Typed GPU dispatch for out-of-place modulo. Delegates to the shared std::visit
    // machinery in cuArithmeticDispatch.cuh (op_code 4 = Mod), which mirrors the CPU
    // design: dispatch on the ordinary Cytnx value types and map to the CUDA-native
    // representation only at the kernel boundary (#1013). Output dtype is the plain
    // type_promote(L, R) (integral -> %, floating -> fmod/fmodf).
    void cuMod_dispatch(boost::intrusive_ptr<Storage_base> &out,
                        boost::intrusive_ptr<Storage_base> &Lin,
                        boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                        const std::vector<cytnx_uint64> &shape,
                        const std::vector<cytnx_uint64> &invmapper_L,
                        const std::vector<cytnx_uint64> &invmapper_R) {
      // Modulo is undefined for complex operands (matches the CPU ModOp and the old
      // cuMod_internal_* host functions, which all error out for complex). Reject
      // here so the (compiled-but-unreachable) complex branch in ApplyGpuArithOp is
      // never executed.
      cytnx_error_msg(Type.is_complex(Lin->dtype()) || Type.is_complex(Rin->dtype()),
                      "[cuMod] Cannot mod complex numbers%s", "\n");

      cuArithmeticDispatchGPU<4>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
