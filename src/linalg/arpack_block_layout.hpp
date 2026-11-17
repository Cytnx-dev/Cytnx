#pragma once
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include "Device.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace linalg {

    // The ARPACK drivers (Lanczos, Arnoldi) see a block UniTensor as one flat vector: the blocks
    // of the start vector, concatenated in the order that tensor stores them. That order is the
    // layout of the whole iteration, and the order in which another tensor of the same space
    // stores its blocks is unrelated to it: a contraction is free to emit the sectors in any
    // order (#738). This function therefore writes the blocks of `output` into `data` at the
    // offsets their sectors have in `reference`, identifying a block by its qindices rather than
    // by its position.
    //
    // A sector of `reference` that `output` does not store is a zero block. A block of `output`
    // whose sector `reference` does not store lies outside the space the iteration runs in and
    // is an error, as are bonds that differ from those of `reference` (the qindices would then
    // index different sectors).
    //
    // `output` must be contiguous and, for a fermionic tensor, have its signflips applied.
    // `data` is a host buffer holding as many elements as `reference` does.
    template <typename T>
    void copy_blocks_in_reference_layout(const UniTensor &reference, const UniTensor &output,
                                         T *data, const char *solver) {
      if (output.uten_type() != reference.uten_type()) {
        cytnx_error_msg(true,
                        "[ERROR][%s] LinOp.matvec(UniTensor) returned a UniTensor of another "
                        "type than the input.\n",
                        solver);
      }
      if (output.bonds() != reference.bonds()) {
        cytnx_error_msg(true,
                        "[ERROR][%s] LinOp.matvec(UniTensor) returned a UniTensor whose bonds "
                        "differ from those of the input.\n",
                        solver);
      }
      const std::vector<Tensor> &reference_blocks = reference.get_blocks_();
      const std::vector<Tensor> &output_blocks = output.get_blocks_();
      const auto &reference_qindices = reference.get_itoi();
      const auto &output_qindices = output.get_itoi();

      std::size_t matched_blocks = 0;
      for (std::size_t ref_idx = 0; ref_idx < reference_blocks.size(); ++ref_idx) {
        const std::size_t block_size = reference_blocks[ref_idx].storage().size();
        const auto found =
          std::find(output_qindices.begin(), output_qindices.end(), reference_qindices[ref_idx]);
        if (found == output_qindices.end()) {
          std::fill(data, data + block_size, T(0));
        } else {
          const Tensor &block = output_blocks[found - output_qindices.begin()];
          if (block.shape() != reference_blocks[ref_idx].shape()) {
            cytnx_error_msg(true,
                            "[ERROR][%s] LinOp.matvec(UniTensor) returned a block whose shape "
                            "differs from the block of the same sector in the input.\n",
                            solver);
          }
          const T *block_data = block.template ptr_as<T>();
          if (block.device() == Device.cpu) {
            std::memcpy(data, block_data, block_size * sizeof(T));
          } else {
#ifdef UNI_GPU
            checkCudaErrors(
              cudaMemcpy(data, block_data, block_size * sizeof(T), cudaMemcpyDeviceToHost));
#endif
          }
          ++matched_blocks;
        }
        data += block_size;
      }
      if (matched_blocks != output_blocks.size()) {
        cytnx_error_msg(true,
                        "[ERROR][%s] LinOp.matvec(UniTensor) returned a block in a sector that "
                        "the input does not have.\n",
                        solver);
      }
    }

  }  // namespace linalg
}  // namespace cytnx
