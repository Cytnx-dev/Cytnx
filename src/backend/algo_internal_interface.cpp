#include "algo_internal_interface.hpp"

// Macro to assign a device-specific sort function to a function table.
#define CYTNX_ASSIGN_SORT_FUNC(Device, Table, DataType) \
  Table[Type.DataType] = Device##Sort_internal_##DataType

// Macro to assign all supported data type sort functions for a specific device.
#define CYTNX_ASSIGN_ALL_DTYPE_SORT_FUNC(Device, Table) \
  CYTNX_ASSIGN_SORT_FUNC(Device, Table, ComplexDouble); \
  CYTNX_ASSIGN_SORT_FUNC(Device, Table, ComplexFloat);  \
  CYTNX_ASSIGN_SORT_FUNC(Device, Table, Double);        \
  CYTNX_ASSIGN_SORT_FUNC(Device, Table, Float);         \
  CYTNX_ASSIGN_SORT_FUNC(Device, Table, Uint64);        \
  CYTNX_ASSIGN_SORT_FUNC(Device, Table, Int64);         \
  CYTNX_ASSIGN_SORT_FUNC(Device, Table, Uint32);        \
  CYTNX_ASSIGN_SORT_FUNC(Device, Table, Int32);         \
  CYTNX_ASSIGN_SORT_FUNC(Device, Table, Uint16);        \
  CYTNX_ASSIGN_SORT_FUNC(Device, Table, Int16);         \
  CYTNX_ASSIGN_SORT_FUNC(Device, Table, Bool);

namespace cytnx {
  namespace algo_internal {

    algo_internal_interface aii;

    algo_internal_interface::algo_internal_interface() {
      Sort_ii.assign(N_Type, nullptr);
#ifdef UNI_GPU
      cuSort_ii.assign(N_Type, nullptr);
#endif
      CYTNX_ASSIGN_ALL_DTYPE_SORT_FUNC(/*cpu*/, Sort_ii)

#ifdef UNI_GPU
      CYTNX_ASSIGN_ALL_DTYPE_SORT_FUNC(cuda, cuSort_ii)
#endif
    }

  }  // namespace algo_internal
}  // namespace cytnx
