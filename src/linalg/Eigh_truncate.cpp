#include "linalg.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "Accessor.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
#include "block_truncation_helpers.hpp"

namespace cytnx {
  namespace linalg {

    namespace {

      // Convert only after bounding by the available spectrum in the unsigned domain.
      std::size_t bounded_dimension(std::uint64_t requested, std::size_t available) {
        return requested < available ? static_cast<std::size_t>(requested) : available;
      }

      // The eigenvalues of one block as host doubles (Eigh lists them in ascending order).
      std::vector<double> block_values(const Tensor &e_block) {
        Tensor as_double = e_block.astype(Type.Double);
        std::vector<double> values(as_double.shape()[0]);
        for (std::size_t i = 0; i < values.size(); i++) {
          values[i] = as_double.storage().at<double>(i);
        }
        return values;
      }

      // Indices of a block's eigenvalues ordered by decreasing magnitude; ties keep the
      // ascending-value order Eigh produced, so the selection is deterministic.
      std::vector<std::size_t> magnitude_order(const std::vector<double> &values) {
        std::vector<std::size_t> order(values.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(), [&values](std::size_t a, std::size_t b) {
          return std::abs(values[a]) > std::abs(values[b]);
        });
        return order;
      }

      // The gather accessor for the kept eigenpairs of a block, in Eigh's ascending order.
      Accessor kept_accessor(const std::vector<std::size_t> &order, std::size_t kdim) {
        std::vector<cytnx_uint64> kept(order.begin(), order.begin() + kdim);
        std::sort(kept.begin(), kept.end());
        return Accessor(kept);
      }

      // The magnitudes that fall below the cut, ascending, as a Tensor of the eigenvalue dtype,
      // in the form BuildBlockDiscardedSingularValues expects.
      Tensor dropped_magnitudes(std::vector<double> dropped, unsigned int dtype, int device) {
        std::sort(dropped.begin(), dropped.end());
        Tensor out({static_cast<cytnx_uint64>(dropped.size())}, Type.Double, Device.cpu);
        for (std::size_t i = 0; i < dropped.size(); i++) out.storage().at<double>(i) = dropped[i];
        return out.astype(dtype).to(device);
      }

      // Chooses the magnitude cut from the candidates that are not guaranteed by a per-block
      // floor: keeps the `keep_dim` largest magnitudes, drops those below `err` down to
      // `min_dim`, and keeps an exact degeneracy at the cut entirely (as Svd_truncate does).
      // Returns the cut and fills `dropped` with the magnitudes below it.
      double magnitude_cut(std::vector<double> candidates, std::size_t keep_dim,
                           std::size_t min_dim, double err, std::vector<double> &dropped) {
        std::sort(candidates.begin(), candidates.end());
        const std::size_t n_cand = candidates.size();
        double cut;
        if (keep_dim == 0) {
          cut = std::numeric_limits<double>::infinity();
        } else {
          if (keep_dim > n_cand) keep_dim = n_cand;
          cut = candidates[n_cand - keep_dim];
          while (cut < err && keep_dim > min_dim) {
            keep_dim--;
            if (keep_dim == 0) {
              cut = std::numeric_limits<double>::infinity();
              break;
            }
            cut = candidates[n_cand - keep_dim];
          }
        }
        for (double value : candidates) {
          if (value < cut) dropped.push_back(value);
        }
        return cut;
      }

      void eigh_truncate_dense_ut_internal(std::vector<UniTensor> &output, const UniTensor &input,
                                           std::uint64_t keepdim, double err, bool return_vectors,
                                           unsigned int return_err, std::uint64_t mindim) {
        output = linalg::Eigh(input, return_vectors, false);
        const Tensor e_full = output[0].get_block_();
        const std::vector<double> values = block_values(e_full);
        const std::vector<std::size_t> order = magnitude_order(values);
        const std::size_t n = values.size();

        std::vector<double> candidates(n);
        for (std::size_t i = 0; i < n; i++) candidates[i] = std::abs(values[i]);
        std::vector<double> dropped;
        const std::size_t min_dim = bounded_dimension(std::max<std::uint64_t>(mindim, 1), n);
        const double cut =
          magnitude_cut(candidates, bounded_dimension(keepdim, n), min_dim, err, dropped);
        std::size_t kdim = 0;
        while (kdim < n && std::abs(values[order[kdim]]) >= cut) kdim++;

        const Accessor kept = kept_accessor(order, kdim);
        Tensor e_kept = e_full.get({kept});
        UniTensor e;
        Bond eigen_bond(kdim);
        e.Init({eigen_bond, eigen_bond}, output[0].labels(), 1, e_kept.dtype(), e_kept.device(),
               true);
        e.put_block_(e_kept);
        output[0] = e;

        if (return_vectors) {
          const UniTensor &full_vectors = output[1];
          std::vector<Accessor> acs(full_vectors.rank(), Accessor::all());
          acs.back() = kept;
          UniTensor vectors(full_vectors.get_block_().get(acs), false, full_vectors.rowrank());
          vectors.set_labels(full_vectors.labels());
          output[1] = vectors;
        }
        if (return_err) {
          output.push_back(BuildBlockDiscardedSingularValues(
            dropped_magnitudes(dropped, e_full.dtype(), e_full.device()), dropped.size(),
            return_err));
        }
      }

      void eigh_truncate_block_ut_internal(std::vector<UniTensor> &output, const UniTensor &input,
                                           std::uint64_t keepdim,
                                           std::vector<cytnx_uint64> min_blockdim, double err,
                                           bool return_vectors, unsigned int return_err,
                                           std::uint64_t mindim) {
        cytnx_error_msg(min_blockdim.empty(),
                        "[ERROR][Eigh_truncate] min_blockdim must not be empty; use the overload "
                        "without min_blockdim if no per-block floor is needed.%s",
                        "\n");
        output = linalg::Eigh(input, return_vectors, false);
        // the return_err entry is pushed below while e and vectors are referenced: no reallocation
        output.reserve(output.size() + 1);
        UniTensor &e = output[0];
        const std::size_t nblocks = e.Nblocks();
        if (min_blockdim.size() == 1) min_blockdim.resize(nblocks, min_blockdim.front());
        cytnx_error_msg(min_blockdim.size() != nblocks,
                        "[ERROR][Eigh_truncate] min_blockdim must have the same number of elements "
                        "as blocks in the eigenvalue UniTensor%s",
                        "\n");

        // 1) per block: the eigenvalues, their magnitude order and the clamped floor; the
        //    magnitudes beyond the floor are the candidates for the global cut.
        std::vector<std::vector<double>> values(nblocks);
        std::vector<std::vector<std::size_t>> order(nblocks);
        std::vector<std::size_t> floors(nblocks);
        std::vector<double> candidates;
        std::size_t floor_count = 0;
        for (std::size_t b = 0; b < nblocks; b++) {
          values[b] = block_values(e.get_block_(b));
          order[b] = magnitude_order(values[b]);
          floors[b] = bounded_dimension(min_blockdim[b], values[b].size());
          floor_count += floors[b];
          for (std::size_t i = floors[b]; i < values[b].size(); i++) {
            candidates.push_back(std::abs(values[b][order[b][i]]));
          }
        }
        // Bound public unsigned dimensions before subtracting floors. Saturating subtraction
        // preserves floor priority without narrowing or wrapping for UINT64_MAX arguments.
        const std::size_t total_count = floor_count + candidates.size();
        const std::size_t keep_count = bounded_dimension(keepdim, total_count);
        const std::size_t min_count =
          bounded_dimension(std::max<std::uint64_t>(mindim, 1), total_count);
        const std::size_t keep_dim = keep_count > floor_count ? keep_count - floor_count : 0;
        const std::size_t min_dim = min_count > floor_count ? min_count - floor_count : 0;

        // 2) the global magnitude cut and the discarded magnitudes.
        std::vector<double> dropped;
        double cut = std::numeric_limits<double>::infinity();
        if (!candidates.empty()) cut = magnitude_cut(candidates, keep_dim, min_dim, err, dropped);
        if (return_err) {
          output.push_back(BuildBlockDiscardedSingularValues(
            dropped_magnitudes(dropped, e.dtype(), e.device()), dropped.size(), return_err));
        }

        // 3) truncate the eigenvalue UniTensor block by block (gather keeps Eigh's order).
        std::vector<cytnx_uint64> new_dims;
        std::vector<std::size_t> keep_dims(nblocks);
        std::vector<std::size_t> new_qid(nblocks);
        std::vector<std::vector<cytnx_uint64>> new_itoi;
        std::vector<Accessor> kept(nblocks, Accessor::all());
        std::vector<cytnx_uint64> to_be_removed;
        std::size_t tot_dim = 0;
        for (std::size_t b = 0; b < nblocks; b++) {
          std::size_t kdim = floors[b];
          while (kdim < values[b].size() && std::abs(values[b][order[b][kdim]]) >= cut) kdim++;
          keep_dims[b] = kdim;
          if (kdim == 0) {
            to_be_removed.push_back(b);
            continue;
          }
          new_qid[b] = new_dims.size();
          new_itoi.push_back({new_dims.size(), new_dims.size()});
          new_dims.push_back(kdim);
          tot_dim += kdim;
          if (kdim != values[b].size()) {
            kept[b] = kept_accessor(order[b], kdim);
            e.get_blocks_()[b] = e.get_blocks_()[b].get({kept[b]});
          }
        }
        e.get_itoi() = new_itoi;
        if (!to_be_removed.empty()) {
          vec_erase_(e.get_blocks_(), to_be_removed);
          vec_erase_(e._impl->_bonds[0]._impl->_qnums, to_be_removed);
          if (input.uten_type() == UTenType.BlockFermionic) {
            static_cast<BlockFermionicUniTensor *>(e._impl.get())->erase_signflip_(to_be_removed);
          }
        }
        e._impl->_bonds[0]._impl->_degs = new_dims;
        e._impl->_bonds[0]._impl->_dim = tot_dim;
        e._impl->_bonds[1] = e._impl->_bonds[0].redirect();

        // 4) the eigenvector UniTensor: its last leg is the eigen bond.
        if (return_vectors) {
          UniTensor &vectors = output[1];
          to_be_removed.clear();
          vectors._impl->_bonds.back() = e._impl->_bonds[1].clone();
          std::vector<Accessor> acs(vectors.rank(), Accessor::all());
          for (std::size_t b = 0; b < vectors.Nblocks(); b++) {
            const std::size_t eb = vectors.get_qindices(b).back();
            if (keep_dims[eb] == 0) {
              to_be_removed.push_back(b);
              continue;
            }
            if (keep_dims[eb] != vectors.get_blocks_()[b].shape().back()) {
              acs.back() = kept[eb];
              vectors.get_blocks_()[b] = vectors.get_blocks_()[b].get(acs);
            }
            vectors.get_qindices(b).back() = new_qid[eb];
          }
          if (!to_be_removed.empty()) {
            vec_erase_(vectors.get_itoi(), to_be_removed);
            vec_erase_(vectors.get_blocks_(), to_be_removed);
            if (input.uten_type() == UTenType.BlockFermionic) {
              static_cast<BlockFermionicUniTensor *>(vectors._impl.get())
                ->erase_signflip_(to_be_removed);
            }
          }
        }
      }

      void check_eigh_truncate_input(const UniTensor &input, std::uint64_t keepdim) {
        cytnx_error_msg(
          input.rank() <= 1,
          "[ERROR][Eigh_truncate] Input UniTensor should have rank>1, but rank is %d\n",
          input.rank());
        cytnx_error_msg(
          input.rowrank() < 1,
          "[ERROR][Eigh_truncate] Input UniTensor should have rowrank>0, but rowrank is %d\n",
          input.rowrank());
        cytnx_error_msg(input.rowrank() >= input.rank(),
                        "[ERROR][Eigh_truncate] Input UniTensor should have rowrank<rank, but "
                        "rowrank is %d and rank is %d\n",
                        input.rowrank(), input.rank());
        cytnx_error_msg(input.is_diag(),
                        "[ERROR][Eigh_truncate] Input UniTensor is diagonal, so Eigh_truncate is "
                        "trivial and not supported. Use other manipulation.%s",
                        "\n");
        cytnx_error_msg(keepdim < 1, "[ERROR][Eigh_truncate] keepdim must be >=1%s", "\n");
      }

    }  // namespace

    std::vector<UniTensor> Eigh_truncate(const UniTensor &input, cytnx_uint64 keepdim, double err,
                                         bool return_vectors, unsigned int return_err,
                                         cytnx_uint64 mindim) {
      check_eigh_truncate_input(input, keepdim);
      std::vector<UniTensor> output;
      if (input.uten_type() == UTenType.Dense) {
        eigh_truncate_dense_ut_internal(output, input, keepdim, err, return_vectors, return_err,
                                        mindim);
      } else if (input.uten_type() == UTenType.Block ||
                 input.uten_type() == UTenType.BlockFermionic) {
        eigh_truncate_block_ut_internal(output, input, keepdim, {0}, err, return_vectors,
                                        return_err, mindim);
      } else {
        cytnx_error_msg(true, "[ERROR][Eigh_truncate] UniTensor type '%s' not supported\n",
                        input.uten_type_str().c_str());
      }
      return output;
    }

    std::vector<UniTensor> Eigh_truncate(const UniTensor &input, cytnx_uint64 keepdim,
                                         std::vector<cytnx_uint64> min_blockdim, double err,
                                         bool return_vectors, unsigned int return_err,
                                         cytnx_uint64 mindim) {
      check_eigh_truncate_input(input, keepdim);
      std::vector<UniTensor> output;
      if (input.uten_type() == UTenType.Dense) {
        cytnx_error_msg(
          min_blockdim.size() != 1,
          "[ERROR][Eigh_truncate] min_blockdim must have one element for dense UniTensor%s", "\n");
        eigh_truncate_dense_ut_internal(output, input, keepdim, err, return_vectors, return_err,
                                        std::max(mindim, min_blockdim[0]));
      } else if (input.uten_type() == UTenType.Block ||
                 input.uten_type() == UTenType.BlockFermionic) {
        eigh_truncate_block_ut_internal(output, input, keepdim, min_blockdim, err, return_vectors,
                                        return_err, mindim);
      } else {
        cytnx_error_msg(true, "[ERROR][Eigh_truncate] UniTensor type '%s' not supported\n",
                        input.uten_type_str().c_str());
      }
      return output;
    }

  }  // namespace linalg
}  // namespace cytnx
