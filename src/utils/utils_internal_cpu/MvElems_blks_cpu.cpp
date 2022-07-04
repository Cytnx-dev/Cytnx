
#include "utils/vec_unique.hpp"
#include "utils/vec_concatenate.hpp"
#include "utils/vec_map.hpp"
#include "utils/cartesian.hpp"
#include "Tensor.hpp"
// #include "utils/utils_internal_interface.hpp"
#include <map>

using namespace std;

namespace cytnx {
  namespace utils_internal {
    template <class T>
    void _moving_elem(vector<Tensor> &dest_blocks, const vector<Tensor> &src_blocks,
                      const vector<cytnx_uint64> &src_shape,
                      const vector<vector<cytnx_uint64>> &src_inner2outer_row,
                      const vector<vector<cytnx_uint64>> &src_inner2outer_col,
                      map<cytnx_uint64, pair<cytnx_uint64, cytnx_uint64>> &dest_outer2inner_row,
                      map<cytnx_uint64, pair<cytnx_uint64, cytnx_uint64>> &dest_outer2inner_col,
                      const vector<cytnx_uint64> &mapper, const vector<cytnx_uint64> &inv_mapper,
                      const cytnx_uint64 &src_inner_rowrank, const cytnx_uint64 &dest_rowrank) {
      const vector<cytnx_uint64> dest_shape = src_shape;

      for (unsigned int b = 0; b < src_blocks.size(); b++) {
        vector<cytnx_uint64> oldshape = vec_map(src_shape, inv_mapper);
        vector<cytnx_uint64> acc_in_old(src_inner_rowrank),
          acc_out_old(oldshape.size() - src_inner_rowrank);
        acc_out_old[acc_out_old.size() - 1] = 1;
        acc_in_old[acc_in_old.size() - 1] = 1;
        for (unsigned int s = 0; s < acc_out_old.size() - 1; s++) {
          acc_out_old[acc_out_old.size() - 2 - s] =
            oldshape.back() * acc_out_old[acc_out_old.size() - 1 - s];
          oldshape.pop_back();
        }
        oldshape.pop_back();
        for (unsigned int s = 0; s < acc_in_old.size() - 1; s++) {
          acc_in_old[acc_in_old.size() - 2 - s] =
            oldshape.back() * acc_in_old[acc_in_old.size() - 1 - s];
          oldshape.pop_back();
        }
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned int elem = 0; elem < src_blocks[b].storage().size(); elem++) {
          unsigned int i = elem / src_blocks[b].shape()[1];
          unsigned int j = elem % src_blocks[b].shape()[1];
          // decompress
          vector<cytnx_uint64> tfidx =
            vec_concatenate(c2cartesian(src_inner2outer_row[b][i], acc_in_old),
                            c2cartesian(src_inner2outer_col[b][j], acc_out_old));
          tfidx = vec_map(tfidx, mapper);  // convert to new index

          // caluclate new row col index:
          cytnx_uint64 new_row = 0, new_col = 0;
          cytnx_uint64 buff = 1;
          for (unsigned int k = 0; k < src_shape.size() - dest_rowrank; k++) {
            new_col += buff * tfidx.back();
            tfidx.pop_back();
            buff *= dest_shape[dest_shape.size() - 1 - k];
          }
          buff = 1;
          for (unsigned int k = 0; k < dest_rowrank; k++) {
            new_row += buff * tfidx.back();
            tfidx.pop_back();
            buff *= dest_shape[dest_rowrank - 1 - k];
          }

          auto dest_mem =
            dest_blocks[dest_outer2inner_row[new_row].first]._impl->storage()._impl->Mem;
          auto src_mem = src_blocks[b]._impl->storage()._impl->Mem;

          cytnx_int64 dest_idx = (dest_outer2inner_row[new_row].second) *
                                   (dest_blocks[dest_outer2inner_row[new_row].first].shape()[1]) +
                                 dest_outer2inner_col[new_col].second;
          cytnx_int64 src_idx = i * src_blocks[b].shape()[1] + j;

          static_cast<T *>(dest_mem)[dest_idx] = static_cast<T *>(src_mem)[src_idx];
        }  // traversal elements in given block b
      }  // end b loop
    }  // end mvelem

    void blocks_mvelems_d(vector<Tensor> &dest_blocks, const vector<Tensor> &src_blocks,
                          const vector<cytnx_uint64> &src_shape,
                          const vector<vector<cytnx_uint64>> &src_inner2outer_row,
                          const vector<vector<cytnx_uint64>> &src_inner2outer_col,
                          map<cytnx_uint64, pair<cytnx_uint64, cytnx_uint64>> &dest_outer2inner_row,
                          map<cytnx_uint64, pair<cytnx_uint64, cytnx_uint64>> &dest_outer2inner_col,
                          const vector<cytnx_uint64> &mapper,
                          const vector<cytnx_uint64> &inv_mapper,
                          const cytnx_uint64 &src_inner_rowrank, const cytnx_uint64 &dest_rowrank) {
      _moving_elem<cytnx_double>(dest_blocks, src_blocks, src_shape, src_inner2outer_row,
                                 src_inner2outer_col, dest_outer2inner_row, dest_outer2inner_col,
                                 mapper, inv_mapper, src_inner_rowrank, dest_rowrank);
    }  // end blocks_mvelems

  }  // namespace utils_internal
}  // namespace cytnx

// void  blocks_mvelems_f(...){
//     _moving_elem<cytnx_float>(...);
// }
// void blocks_mvelems_u64(...){
//     _moving_elem<cytnx_uint64>(...);
// }
// void  blocks_mvelems_u32(...){
//     _moving_elem<cytnx_uint32>(...);
// }
// void blocks_mvelems_u16(...){
//     _moving_elem<cytnx_uint16>(...);
// }
// void  blocks_mvelems_i64(...){
//     _moving_elem<cytnx_int64>(...);
// }
// void blocks_mvelems_i32(...){
//     _moving_elem<cytnx_int32>(...);
// }
// void  blocks_mvelems_i16(...){
//     _moving_elem<cytnx_int16>(...);
// }
// void  blocks_mvelems_cd(...){
//     _moving_elem<cytnx_complex128>(...);
// }
// void  blocks_mvelems_cf(...){
//     _moving_elem<cytnx_complex64>(...);
// }
// void  blocks_mvelems_b(...){
//     _moving_elem<cytnx_bool>(...);
// }

// ---------- in SparseUniTensor.cpp ----------

// boost::intrusive_ptr<UniTensor_base> SparseUniTensor::contiguous(){
//     if(this->is_contiguous()){
//         boost::intrusive_ptr<UniTensor_base> out(this);
//         return out;
//     }else{
//         // make new instance
//         SparseUniTensor* tmp = new SparseUniTensor();
//         tmp->Init(this->_bonds,this->_labels,this->_rowrank,this->dtype(),this->device(),this->_is_diag);
//         if(src->_is_diag){
//             tmp->_blocks[b] = this->_blocks[b];
//         }else{
//             utils_internal::uii.blocks_mvelems_ii[this->dtype()](tmp->_blocks, this->_blocks,
//             this->shape,  this->_inner2outer_row, this->_inner2outer_col, tmp->_outer2inner_row,
//             tmp->_outer2inner_col, this->_mapper, this->_inv_mapper, this->_inner_rowrank,
//             tmp->_rowrank);
//         }
//         //transform to a intr_ptr.
//         boost::intrusive_ptr<UniTensor_base> out(tmp);
//         return out;
//     }
// }
