#include "algo_internal_interface.hpp"

using namespace std;

namespace cytnx {
  namespace algo_internal {

    algo_internal_interface aii;

    algo_internal_interface::algo_internal_interface() {
      Sort_ii.assign(N_Type, NULL);
#ifdef UNI_GPU
      cuSort_ii.assign(N_Type, NULL);
#endif

      Sort_ii[Type.ComplexDouble] = Sort_internal_cd;
      Sort_ii[Type.ComplexFloat] = Sort_internal_cf;
      Sort_ii[Type.Double] = Sort_internal_d;
      Sort_ii[Type.Float] = Sort_internal_f;
      Sort_ii[Type.Uint64] = Sort_internal_u64;
      Sort_ii[Type.Int64] = Sort_internal_i64;
      Sort_ii[Type.Uint32] = Sort_internal_u32;
      Sort_ii[Type.Int32] = Sort_internal_i32;
      Sort_ii[Type.Uint16] = Sort_internal_u16;
      Sort_ii[Type.Int16] = Sort_internal_i16;
      Sort_ii[Type.Bool] = Sort_internal_b;

#ifdef UNI_GPU
      cuSort_ii[Type.ComplexDouble] = cuSort_internal_cd;
      cuSort_ii[Type.ComplexFloat] = cuSort_internal_cf;
      cuSort_ii[Type.Double] = cuSort_internal_d;
      cuSort_ii[Type.Float] = cuSort_internal_f;
      cuSort_ii[Type.Uint64] = cuSort_internal_u64;
      cuSort_ii[Type.Int64] = cuSort_internal_i64;
      cuSort_ii[Type.Uint32] = cuSort_internal_u32;
      cuSort_ii[Type.Int32] = cuSort_internal_i32;
      cuSort_ii[Type.Uint16] = cuSort_internal_u16;
      cuSort_ii[Type.Int16] = cuSort_internal_i16;
      cuSort_ii[Type.Bool] = cuSort_internal_b;
#endif
    }

  }  // namespace algo_internal
}  // namespace cytnx
