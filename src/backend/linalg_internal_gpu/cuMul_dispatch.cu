#include "cuMul_internal.hpp"

#include "cuArithmeticDispatch.cuh"

namespace cytnx {

  namespace linalg_internal {

    // Typed GPU dispatch for out-of-place multiplication. Delegates to the shared
    // std::visit machinery in cuArithmeticDispatch.cuh (op_code 1 = Mul), which
    // mirrors the CPU design: dispatch on the ordinary Cytnx value types and map
    // to the CUDA-native representation only at the kernel boundary (#1013).
    void cuMul_dispatch(boost::intrusive_ptr<Storage_base> &out,
                        boost::intrusive_ptr<Storage_base> &Lin,
                        boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                        const std::vector<cytnx_uint64> &shape,
                        const std::vector<cytnx_uint64> &invmapper_L,
                        const std::vector<cytnx_uint64> &invmapper_R) {
      cuArithmeticDispatchGPU<1>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
