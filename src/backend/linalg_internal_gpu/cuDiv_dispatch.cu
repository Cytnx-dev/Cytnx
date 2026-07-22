#include "cuDiv_internal.hpp"

#include "cuArithmeticDispatch.cuh"

namespace cytnx {

  namespace linalg_internal {

    // Typed GPU dispatch for out-of-place division. Delegates to the shared
    // std::visit machinery in cuArithmeticDispatch.cuh (op_code 3 = Div), which
    // mirrors the CPU design: dispatch on the ordinary Cytnx value types and map
    // to the CUDA-native representation only at the kernel boundary (#1013).
    //
    // NUMERICAL CHANGE (#941/#1013): op_code 3 is *true division* -- the output
    // dtype is make_floating_point_t<type_promote<TL,TR>>, so integer/integer
    // divides in floating point (e.g. Int64 / Int64 -> Double), matching the CPU
    // path. This replaces the legacy GPU integer division (type_promote_gpu_t).
    // The caller (linalg::Div) allocates `out` with the floating dtype; see the
    // collapse of div_output_dtype in src/linalg/Div.cpp.
    void cuDiv_dispatch(boost::intrusive_ptr<Storage_base> &out,
                        boost::intrusive_ptr<Storage_base> &Lin,
                        boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                        const std::vector<cytnx_uint64> &shape,
                        const std::vector<cytnx_uint64> &invmapper_L,
                        const std::vector<cytnx_uint64> &invmapper_R) {
      cuArithmeticDispatchGPU<3>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
