#include "linalg.hpp"

#include <algorithm>
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
      std::vector<cytnx_uint64> magnitude_order(const std::vector<double> &values) {
        std::vector<cytnx_uint64> order(values.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&values](cytnx_uint64 a, cytnx_uint64 b) {
          return std::abs(values[a]) > std::abs(values[b]);
        });
        return order;
      }

      // The gather accessor for the kept eigenpairs of a block, in Eigh's ascending order.
      Accessor kept_accessor(const std::vector<cytnx_uint64> &order, cytnx_uint64 kdim) {
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
      double magnitude_cut(std::vector<double> candidates, cytnx_int64 keep_dim,
                           cytnx_int64 min_dim, double err, std::vector<double> &dropped) {
        std::sort(candidates.begin(), candidates.end());
        const cytnx_int64 n_cand = static_cast<cytnx_int64>(candidates.size());
        double cut;
        if (keep_dim <= 0) {
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

      void Eigh_truncate_Dense_UT_internal(std::vector<UniTensor> &outCyT, const UniTensor &Tin,
                                           cytnx_uint64 keepdim, double err, bool is_V,
                                           unsigned int return_err, cytnx_uint64 mindim) {
        outCyT = linalg::Eigh(Tin, is_V, false);
        const Tensor e_full = outCyT[0].get_block_();
        const std::vector<double> values = block_values(e_full);
        const std::vector<cytnx_uint64> order = magnitude_order(values);
        const cytnx_uint64 n = values.size();

        std::vector<double> candidates(n);
        for (cytnx_uint64 i = 0; i < n; i++) candidates[i] = std::abs(values[i]);
        std::vector<double> dropped;
        const cytnx_int64 min_dim = static_cast<cytnx_int64>(mindim < 1 ? 1 : mindim);
        const double cut =
          magnitude_cut(candidates, static_cast<cytnx_int64>(keepdim), min_dim, err, dropped);
        cytnx_uint64 kdim = 0;
        while (kdim < n && std::abs(values[order[kdim]]) >= cut) kdim++;
        if (kdim == 0) kdim = 1;  // Svd_truncate's rule: at least one value is always kept

        const Accessor kept = kept_accessor(order, kdim);
        Tensor e_kept = e_full.get({kept});
        UniTensor e;
        Bond eigen_bond(kdim);
        e.Init({eigen_bond, eigen_bond}, outCyT[0].labels(), 1, e_kept.dtype(), e_kept.device(),
               true);
        e.put_block_(e_kept);
        outCyT[0] = e;

        if (is_V) {
          const UniTensor &V_full = outCyT[1];
          std::vector<Accessor> acs(V_full.rank(), Accessor::all());
          acs.back() = kept;
          UniTensor V(V_full.get_block_().get(acs), false, V_full.rowrank());
          V.set_labels(V_full.labels());
          outCyT[1] = V;
        }
        if (return_err) {
          outCyT.push_back(BuildBlockDiscardedSingularValues(
            dropped_magnitudes(dropped, e_full.dtype(), e_full.device()), dropped.size(),
            return_err));
        }
      }

      void Eigh_truncate_Block_UT_internal(std::vector<UniTensor> &outCyT, const UniTensor &Tin,
                                           cytnx_uint64 keepdim,
                                           std::vector<cytnx_uint64> min_blockdim, double err,
                                           bool is_V, unsigned int return_err,
                                           cytnx_uint64 mindim) {
        cytnx_error_msg(min_blockdim.empty(),
                        "[ERROR][Eigh_truncate] min_blockdim must not be empty; use the overload "
                        "without min_blockdim if no per-block floor is needed.%s",
                        "\n");
        outCyT = linalg::Eigh(Tin, is_V, false);
        // the return_err entry is pushed below while e and V are referenced: no reallocation
        outCyT.reserve(outCyT.size() + 1);
        UniTensor &e = outCyT[0];
        const cytnx_uint64 nblocks = e.Nblocks();
        if (min_blockdim.size() == 1) min_blockdim.resize(nblocks, min_blockdim.front());
        cytnx_error_msg(min_blockdim.size() != nblocks,
                        "[ERROR][Eigh_truncate] min_blockdim must have the same number of elements "
                        "as blocks in the eigenvalue UniTensor%s",
                        "\n");

        // 1) per block: the eigenvalues, their magnitude order and the clamped floor; the
        //    magnitudes beyond the floor are the candidates for the global cut.
        std::vector<std::vector<double>> values(nblocks);
        std::vector<std::vector<cytnx_uint64>> order(nblocks);
        std::vector<cytnx_uint64> floors(nblocks);
        std::vector<double> candidates;
        cytnx_int64 keep_dim = static_cast<cytnx_int64>(keepdim);
        cytnx_int64 min_dim = static_cast<cytnx_int64>(mindim < 1 ? 1 : mindim);
        bool any_floor = false;
        for (cytnx_uint64 b = 0; b < nblocks; b++) {
          values[b] = block_values(e.get_block_(b));
          order[b] = magnitude_order(values[b]);
          floors[b] = std::min<cytnx_uint64>(min_blockdim[b], values[b].size());
          if (min_blockdim[b] > 0) any_floor = true;
          keep_dim -= static_cast<cytnx_int64>(floors[b]);
          min_dim -= static_cast<cytnx_int64>(floors[b]);
          for (cytnx_uint64 i = floors[b]; i < values[b].size(); i++) {
            candidates.push_back(std::abs(values[b][order[b][i]]));
          }
        }
        if (!any_floor && min_dim < 1) min_dim = 1;
        if (any_floor && min_dim < 0) min_dim = 0;

        // 2) the global magnitude cut and the discarded magnitudes.
        std::vector<double> dropped;
        double cut = std::numeric_limits<double>::infinity();
        if (!candidates.empty()) cut = magnitude_cut(candidates, keep_dim, min_dim, err, dropped);
        if (return_err) {
          outCyT.push_back(BuildBlockDiscardedSingularValues(
            dropped_magnitudes(dropped, e.dtype(), e.device()), dropped.size(), return_err));
        }

        // 3) truncate the eigenvalue UniTensor block by block (gather keeps Eigh's order).
        std::vector<cytnx_uint64> new_dims;
        std::vector<cytnx_int64> keep_dims(nblocks);
        std::vector<cytnx_int64> new_qid(nblocks);
        std::vector<std::vector<cytnx_uint64>> new_itoi;
        std::vector<Accessor> kept(nblocks, Accessor::all());
        std::vector<cytnx_uint64> to_be_removed;
        cytnx_uint64 tot_dim = 0;
        for (cytnx_uint64 b = 0; b < nblocks; b++) {
          cytnx_uint64 kdim = floors[b];
          while (kdim < values[b].size() && std::abs(values[b][order[b][kdim]]) >= cut) kdim++;
          keep_dims[b] = static_cast<cytnx_int64>(kdim);
          if (kdim == 0) {
            to_be_removed.push_back(b);
            new_qid[b] = -1;
            continue;
          }
          new_qid[b] = static_cast<cytnx_int64>(new_dims.size());
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
          if (Tin.uten_type() == UTenType.BlockFermionic) {
            static_cast<BlockFermionicUniTensor *>(e._impl.get())->erase_signflip_(to_be_removed);
          }
        }
        e._impl->_bonds[0]._impl->_degs = new_dims;
        e._impl->_bonds[0]._impl->_dim = tot_dim;
        e._impl->_bonds[1] = e._impl->_bonds[0].redirect();

        // 4) the eigenvector UniTensor: its last leg is the eigen bond.
        if (is_V) {
          UniTensor &V = outCyT[1];
          to_be_removed.clear();
          V._impl->_bonds.back() = e._impl->_bonds[1].clone();
          std::vector<Accessor> acs(V.rank(), Accessor::all());
          for (cytnx_uint64 b = 0; b < V.Nblocks(); b++) {
            const cytnx_uint64 eb = V.get_qindices(b).back();
            if (keep_dims[eb] == 0) {
              to_be_removed.push_back(b);
              continue;
            }
            if (keep_dims[eb] != static_cast<cytnx_int64>(V.get_blocks_()[b].shape().back())) {
              acs.back() = kept[eb];
              V.get_blocks_()[b] = V.get_blocks_()[b].get(acs);
            }
            V.get_qindices(b).back() = new_qid[eb];
          }
          if (!to_be_removed.empty()) {
            vec_erase_(V.get_itoi(), to_be_removed);
            vec_erase_(V.get_blocks_(), to_be_removed);
            if (Tin.uten_type() == UTenType.BlockFermionic) {
              static_cast<BlockFermionicUniTensor *>(V._impl.get())->erase_signflip_(to_be_removed);
            }
          }
        }
      }

      void check_eigh_truncate_input(const UniTensor &Tin, cytnx_uint64 keepdim) {
        cytnx_error_msg(
          Tin.rank() <= 1,
          "[ERROR][Eigh_truncate] Input UniTensor should have rank>1, but rank is %d\n",
          Tin.rank());
        cytnx_error_msg(
          Tin.rowrank() < 1,
          "[ERROR][Eigh_truncate] Input UniTensor should have rowrank>0, but rowrank is %d\n",
          Tin.rowrank());
        cytnx_error_msg(Tin.rowrank() >= Tin.rank(),
                        "[ERROR][Eigh_truncate] Input UniTensor should have rowrank<rank, but "
                        "rowrank is %d and rank is %d\n",
                        Tin.rowrank(), Tin.rank());
        cytnx_error_msg(Tin.is_diag(),
                        "[ERROR][Eigh_truncate] Input UniTensor is diagonal, so Eigh_truncate is "
                        "trivial and not supported. Use other manipulation.%s",
                        "\n");
        cytnx_error_msg(keepdim < 1, "[ERROR][Eigh_truncate] keepdim must be >=1%s", "\n");
      }

    }  // namespace

    std::vector<UniTensor> Eigh_truncate(const UniTensor &Tin, cytnx_uint64 keepdim, double err,
                                         bool is_V, unsigned int return_err, cytnx_uint64 mindim) {
      check_eigh_truncate_input(Tin, keepdim);
      std::vector<UniTensor> outCyT;
      if (Tin.uten_type() == UTenType.Dense) {
        Eigh_truncate_Dense_UT_internal(outCyT, Tin, keepdim, err, is_V, return_err, mindim);
      } else if (Tin.uten_type() == UTenType.Block || Tin.uten_type() == UTenType.BlockFermionic) {
        Eigh_truncate_Block_UT_internal(outCyT, Tin, keepdim, {0}, err, is_V, return_err, mindim);
      } else {
        cytnx_error_msg(true, "[ERROR][Eigh_truncate] UniTensor type '%s' not supported\n",
                        Tin.uten_type_str().c_str());
      }
      return outCyT;
    }

    std::vector<UniTensor> Eigh_truncate(const UniTensor &Tin, cytnx_uint64 keepdim,
                                         std::vector<cytnx_uint64> min_blockdim, double err,
                                         bool is_V, unsigned int return_err, cytnx_uint64 mindim) {
      check_eigh_truncate_input(Tin, keepdim);
      std::vector<UniTensor> outCyT;
      if (Tin.uten_type() == UTenType.Dense) {
        cytnx_error_msg(
          min_blockdim.size() != 1,
          "[ERROR][Eigh_truncate] min_blockdim must have one element for dense UniTensor%s", "\n");
        Eigh_truncate_Dense_UT_internal(outCyT, Tin, keepdim, err, is_V, return_err,
                                        std::max(mindim, min_blockdim[0]));
      } else if (Tin.uten_type() == UTenType.Block || Tin.uten_type() == UTenType.BlockFermionic) {
        Eigh_truncate_Block_UT_internal(outCyT, Tin, keepdim, min_blockdim, err, is_V, return_err,
                                        mindim);
      } else {
        cytnx_error_msg(true, "[ERROR][Eigh_truncate] UniTensor type '%s' not supported\n",
                        Tin.uten_type_str().c_str());
      }
      return outCyT;
    }

  }  // namespace linalg
}  // namespace cytnx
