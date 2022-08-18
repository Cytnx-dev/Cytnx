#ifndef _H_blocks_mvelems_cpu_
#define _H_blocks_mvelems_cpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "Storage.hpp"

#include "Tensor.hpp"
#include <vector>
#include <map>

using namespace std;

namespace cytnx {
  namespace utils_internal {

    // void _moving_elem(vector<Storage> &dest_blocks, const vector<Storage> &src_blocks,
    //             const vector<vector<cytnx_uint64>> &dest_shapes, const
    //             vector<vector<cytnx_uint64>> &src_shapes, const vector<cytnx_uint64> &src_shape,
    //             const vector<vector<cytnx_uint64>>  &src_inner2outer_row, const
    //             vector<vector<cytnx_uint64>>  &src_inner2outer_col, map<cytnx_uint64,
    //             pair<cytnx_uint64,cytnx_uint64>>  &dest_outer2inner_row,
    //             map<cytnx_uint64,pair<cytnx_uint64, cytnx_uint64>>  &dest_outer2inner_col, const
    //             vector<cytnx_uint64> &mapper, const vector<cytnx_uint64> &inv_mapper, const
    //             cytnx_uint64 &src_inner_rowrank,  const cytnx_uint64 &dest_rowrank);

    void _moving_elem(vector<Tensor> &dest_blocks, const vector<Tensor> &src_blocks,
                      const vector<cytnx_uint64> &src_shape,
                      const vector<vector<cytnx_uint64>> &src_inner2outer_row,
                      const vector<vector<cytnx_uint64>> &src_inner2outer_col,
                      map<cytnx_uint64, pair<cytnx_uint64, cytnx_uint64>> &dest_outer2inner_row,
                      map<cytnx_uint64, pair<cytnx_uint64, cytnx_uint64>> &dest_outer2inner_col,
                      const vector<cytnx_uint64> &mapper, const vector<cytnx_uint64> &inv_mapper,
                      const cytnx_uint64 &src_inner_rowrank, const cytnx_uint64 &dest_rowrank);

    // void blocks_mvelems_d(vector<Storage> &dest_blocks, const vector<Storage> &src_blocks,
    //             const vector<vector<cytnx_uint64>> &dest_shapes, const
    //             vector<vector<cytnx_uint64>> &src_shapes, const vector<cytnx_uint64> &src_shape,
    //             const vector<vector<cytnx_uint64>>  &src_inner2outer_row, const
    //             vector<vector<cytnx_uint64>>  &src_inner2outer_col, map<cytnx_uint64,
    //             pair<cytnx_uint64,cytnx_uint64>>  &dest_outer2inner_row,
    //             map<cytnx_uint64,pair<cytnx_uint64, cytnx_uint64>>  &dest_outer2inner_col, const
    //             vector<cytnx_uint64> &mapper, const vector<cytnx_uint64> &inv_mapper, const
    //             cytnx_uint64 &src_inner_rowrank,  const cytnx_uint64 &dest_rowrank);

    void blocks_mvelems_d(vector<Tensor> &dest_blocks, const vector<Tensor> &src_blocks,
                          const vector<cytnx_uint64> &src_shape,
                          const vector<vector<cytnx_uint64>> &src_inner2outer_row,
                          const vector<vector<cytnx_uint64>> &src_inner2outer_col,
                          map<cytnx_uint64, pair<cytnx_uint64, cytnx_uint64>> &dest_outer2inner_row,
                          map<cytnx_uint64, pair<cytnx_uint64, cytnx_uint64>> &dest_outer2inner_col,
                          const vector<cytnx_uint64> &mapper,
                          const vector<cytnx_uint64> &inv_mapper,
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
#endif
