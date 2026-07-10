#ifndef CYTNX_BACKEND_UTILS_INTERNAL_CPU_BLOCKS_MVELEMS_CPU_H_
#define CYTNX_BACKEND_UTILS_INTERNAL_CPU_BLOCKS_MVELEMS_CPU_H_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "backend/Storage.hpp"

#include "Tensor.hpp"
#include <vector>
#include <map>

namespace cytnx {
  namespace utils_internal {

    // void _moving_elem(std::vector<Storage> &dest_blocks, const std::vector<Storage> &src_blocks,
    //             const std::vector<std::vector<cytnx_uint64>> &dest_shapes, const
    //             std::vector<std::vector<cytnx_uint64>> &src_shapes, const
    //             std::vector<cytnx_uint64> &src_shape, const
    //             std::vector<std::vector<cytnx_uint64>>  &src_inner2outer_row, const
    //             std::vector<std::vector<cytnx_uint64>>  &src_inner2outer_col,
    //             std::map<cytnx_uint64, std::pair<cytnx_uint64,cytnx_uint64>>
    //             &dest_outer2inner_row, std::map<cytnx_uint64,std::pair<cytnx_uint64,
    //             cytnx_uint64>>  &dest_outer2inner_col, const std::vector<cytnx_uint64> &mapper,
    //             const std::vector<cytnx_uint64> &inv_mapper, const cytnx_uint64
    //             &src_inner_rowrank,  const cytnx_uint64 &dest_rowrank);

    void _moving_elem(
      std::vector<Tensor> &dest_blocks, const std::vector<Tensor> &src_blocks,
      const std::vector<cytnx_uint64> &src_shape,
      const std::vector<std::vector<cytnx_uint64>> &src_inner2outer_row,
      const std::vector<std::vector<cytnx_uint64>> &src_inner2outer_col,
      std::map<cytnx_uint64, std::pair<cytnx_uint64, cytnx_uint64>> &dest_outer2inner_row,
      std::map<cytnx_uint64, std::pair<cytnx_uint64, cytnx_uint64>> &dest_outer2inner_col,
      const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &inv_mapper,
      const cytnx_uint64 &src_inner_rowrank, const cytnx_uint64 &dest_rowrank);

    // void blocks_mvelems_d(std::vector<Storage> &dest_blocks, const std::vector<Storage>
    // &src_blocks,
    //             const std::vector<std::vector<cytnx_uint64>> &dest_shapes, const
    //             std::vector<std::vector<cytnx_uint64>> &src_shapes, const
    //             std::vector<cytnx_uint64> &src_shape, const
    //             std::vector<std::vector<cytnx_uint64>>  &src_inner2outer_row, const
    //             std::vector<std::vector<cytnx_uint64>>  &src_inner2outer_col,
    //             std::map<cytnx_uint64, std::pair<cytnx_uint64,cytnx_uint64>>
    //             &dest_outer2inner_row, std::map<cytnx_uint64,std::pair<cytnx_uint64,
    //             cytnx_uint64>>  &dest_outer2inner_col, const std::vector<cytnx_uint64> &mapper,
    //             const std::vector<cytnx_uint64> &inv_mapper, const cytnx_uint64
    //             &src_inner_rowrank,  const cytnx_uint64 &dest_rowrank);

    void blocks_mvelems_d(
      std::vector<Tensor> &dest_blocks, const std::vector<Tensor> &src_blocks,
      const std::vector<cytnx_uint64> &src_shape,
      const std::vector<std::vector<cytnx_uint64>> &src_inner2outer_row,
      const std::vector<std::vector<cytnx_uint64>> &src_inner2outer_col,
      std::map<cytnx_uint64, std::pair<cytnx_uint64, cytnx_uint64>> &dest_outer2inner_row,
      std::map<cytnx_uint64, std::pair<cytnx_uint64, cytnx_uint64>> &dest_outer2inner_col,
      const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &inv_mapper,
      const cytnx_uint64 &src_inner_rowrank, const cytnx_uint64 &dest_rowrank);
    // void blocks_mvelems_f(...)
    // void blocks_mvelems_u64(...)
    // void  blocks_mvelems_u32(...)
    // void blocks_mvelems_u16(...)
    // void  blocks_mvelems_i64(...)
    // void blocks_mvelems_i32(...)
    // void  blocks_mvelems_i16(...)
    // void  blocks_mvelems_cd(...)
    // void  blocks_mvelems_cf(...)
    // void  blocks_mvelems_b(...)

  }  // namespace utils_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_UTILS_INTERNAL_CPU_BLOCKS_MVELEMS_CPU_H_
