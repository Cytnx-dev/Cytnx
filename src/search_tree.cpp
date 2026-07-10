#include "search_tree.hpp"
#include <stack>
#include <queue>
#include <functional>
#include <algorithm>

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  // helper functions
  cytnx_float get_cost(const PseudoUniTensor& t1, const PseudoUniTensor& t2) {
    cytnx_float cost = 1;
    std::vector<cytnx_uint64> shape1 = t1.shape;
    std::vector<cytnx_uint64> shape2 = t2.shape;

    for (cytnx_uint64 i = 0; i < shape1.size(); i++) {
      cost *= shape1[i];
    }
    for (cytnx_uint64 i = 0; i < shape2.size(); i++) {
      cost *= shape2[i];
    }

    // get bond with common label:
    std::vector<std::string> common_lbl;
    std::vector<cytnx_uint64> comm_idx1, comm_idx2;
    vec_intersect_(common_lbl, t1.labels, t2.labels, comm_idx1, comm_idx2);

    for (cytnx_uint64 i = 0; i < comm_idx2.size(); i++) cost /= shape2[comm_idx2[i]];

    return cost + t1.cost + t2.cost;
  }

  PseudoUniTensor pContract(PseudoUniTensor& t1, PseudoUniTensor& t2) {
    PseudoUniTensor t3(0);  // Initialize with index 0

    t3.ID = t1.ID ^ t2.ID;  // XOR of IDs to track contracted tensors
    t3.cost = get_cost(t1, t2);  // Calculate contraction cost

    // Find common labels between t1 and t2
    std::vector<cytnx_uint64> loc1, loc2;
    std::vector<std::string> comm_lbl;
    vec_intersect_(comm_lbl, t1.labels, t2.labels, loc1, loc2);

    // New shape is concatenation of non-contracted dimensions
    t3.shape = vec_concatenate(vec_erase(t1.shape, loc1), vec_erase(t2.shape, loc2));

    // New labels are concatenation of non-contracted labels
    t3.labels = vec_concatenate(vec_erase(t1.labels, loc1), vec_erase(t2.labels, loc2));

    // Set accumulation string using the original accu_str if available
    if (t1.accu_str.empty()) t1.accu_str = std::to_string(t1.tensorIndex);
    if (t2.accu_str.empty()) t2.accu_str = std::to_string(t2.tensorIndex);
    t3.accu_str = "(" + t1.accu_str + "," + t2.accu_str + ")";

    // Set as internal node
    t3.isLeaf = false;
    t3.left = std::make_unique<PseudoUniTensor>(t1);
    t3.right = std::make_unique<PseudoUniTensor>(t2);

    return t3;
  }

  namespace OptimalTreeSolver {
    // Whether two (leaf or already-contracted) nodes share at least one label.
    // Only used to build the initial per-leaf adjacency matrix below -- the
    // pair-selection loop in solve() looks adjacency up in that matrix
    // instead of re-deriving it, so this never runs in that O(n^2)-per-round
    // hot loop.
    bool has_common_label(const PseudoUniTensor& t1, const PseudoUniTensor& t2) {
      for (const auto& l1 : t1.labels) {
        for (const auto& l2 : t2.labels) {
          if (l1 == l2) return true;
        }
      }
      return false;
    }

    // Helper function to find connected components using DFS
    void dfs(std::size_t node, const std::vector<IndexSet>& adjacencyMatrix, IndexSet& visited,
             std::vector<std::size_t>& component) {
      visited.set(node);
      component.push_back(node);

      for (std::size_t i = 0; i < adjacencyMatrix.size(); ++i) {
        if (adjacencyMatrix[node].test(i) && !visited.test(i)) {
          dfs(i, adjacencyMatrix, visited, component);
        }
      }
    }

    // Find connected components in the tensor network
    std::vector<std::vector<std::size_t>> findConnectedComponents(
      const std::vector<IndexSet>& adjacencyMatrix) {
      std::vector<std::vector<std::size_t>> components;
      IndexSet visited;

      for (std::size_t i = 0; i < adjacencyMatrix.size(); ++i) {
        if (!visited.test(i)) {
          std::vector<std::size_t> component;
          dfs(i, adjacencyMatrix, visited, component);
          components.push_back(component);
        }
      }

      return components;
    }

    // A candidate contraction of the two adjacent nodes with adjacency-matrix
    // rows lhs_adj_index and rhs_adj_index (kept lhs_adj_index < rhs_adj_index),
    // tagged with its already-computed get_cost. get_cost depends only on the
    // two nodes' shapes/labels/costs, all fixed once a node is created, so this
    // cost never needs recomputing while the pair survives -- caching it here
    // is what turns the per-round rescan into a heap pop.
    struct PairCandidate {
      cytnx_float cost;
      std::size_t lhs_adj_index;
      std::size_t rhs_adj_index;

      // Order by cost, then by the node indices, so the min-heap reproduces a
      // deterministic tie-break (lowest lhs_adj_index, then rhs_adj_index).
      bool operator>(const PairCandidate& rhs) const {
        if (cost != rhs.cost) return cost > rhs.cost;
        if (lhs_adj_index != rhs.lhs_adj_index) return lhs_adj_index > rhs.lhs_adj_index;
        return rhs_adj_index > rhs.rhs_adj_index;
      }
    };

    std::unique_ptr<PseudoUniTensor> solve(const std::vector<PseudoUniTensor>& tensors,
                                           bool verbose) {
      if (tensors.empty()) {
        return nullptr;
      }

      // A leaf's ID is 1ULL << leaf_index (a 64-bit mask), and the adjacency
      // rows are IndexSet = std::bitset<128>. Contracting a single connected
      // component of k leaves creates k-1 merged nodes, so up to 2k-1 nodes
      // and adjacency indices exist; with k <= 64 that is <= 127 < 128, and
      // the leaf index stays <= 63 so 1ULL << leaf_index does not overflow.
      // More than 64 tensors would shift past the width of both, so reject it.
      cytnx_error_msg(tensors.size() > 64,
                      "[ERROR][OptimalTreeSolver] solve() supports at most 64 tensors, got %d.\n",
                      static_cast<int>(tensors.size()));

      // Build leaf nodes. Each leaf's adj_index is its row in adjacencyMatrix;
      // merged nodes get theirs assigned below, as adjacencyMatrix is extended
      // for them.
      std::vector<std::unique_ptr<PseudoUniTensor>> leaves;
      leaves.reserve(tensors.size());
      for (std::size_t i = 0; i < tensors.size(); ++i) {
        auto leaf = std::make_unique<PseudoUniTensor>(i);
        *leaf = tensors[i];
        leaf->ID = 1ULL << i;
        leaf->adj_index = i;
        leaves.push_back(std::move(leaf));
      }

      const std::size_t n = leaves.size();
      // One adjacency row per node that will ever exist. Leaves fill rows
      // [0, n); each contraction below appends one more row for the merged
      // node it creates (addressed via that node's adj_index), so
      // adjacencyMatrix[...].test(...) stays a valid O(1) lookup for merged
      // nodes too, not just the leaves. This caps the total leaf+merged node
      // count at IndexSet's 128 bits.
      std::vector<IndexSet> adjacencyMatrix(n);

      // Fill adjacency matrix
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
          if (has_common_label(*leaves[i], *leaves[j])) {
            adjacencyMatrix[i].set(j);
            adjacencyMatrix[j].set(i);
          }
        }
      }

      // Find connected components
      std::vector<std::vector<std::size_t>> components = findConnectedComponents(adjacencyMatrix);
      if (verbose && components.size() > 1) {
        std::cout << "Found " << components.size() << " disconnected components" << std::endl;
      }

      // Process each component separately
      std::vector<std::unique_ptr<PseudoUniTensor>> component_results;
      for (const std::vector<std::size_t>& component : components) {
        // Alive nodes of this component, keyed by their adjacencyMatrix row
        // (adj_index) so a candidate pair can be validated and its nodes
        // fetched in O(1). Each node already knows its own row via adj_index,
        // so no separate index-into-a-master-list bookkeeping is needed.
        std::unordered_map<std::size_t, std::unique_ptr<PseudoUniTensor>> alive_nodes;
        alive_nodes.reserve(component.size());
        for (std::size_t leaf_idx : component) {
          std::size_t adj = leaves[leaf_idx]->adj_index;
          alive_nodes.emplace(adj, std::move(leaves[leaf_idx]));
        }

        // Min-heap of candidate contractions. Every adjacent pair's cost is
        // computed exactly once -- when the pair first forms -- and stays
        // valid until one of its nodes is contracted (get_cost is a pure
        // function of the two nodes). A round then costs one O(log) pop plus
        // O(degree) fresh pushes for the new merged node, instead of an
        // O(alive^2) rescan; a candidate whose node has since been contracted
        // is skipped lazily on pop.
        std::priority_queue<PairCandidate, std::vector<PairCandidate>, std::greater<>> candidates;

        // Seed the heap with the component's current adjacent pairs.
        for (std::size_t a = 0; a < component.size(); ++a) {
          for (std::size_t b = a + 1; b < component.size(); ++b) {
            std::size_t a_adj = component[a];
            std::size_t b_adj = component[b];
            if (adjacencyMatrix[a_adj].test(b_adj)) {
              PseudoUniTensor& na = *alive_nodes[a_adj];
              PseudoUniTensor& nb = *alive_nodes[b_adj];
              candidates.push({get_cost(na, nb), std::min(a_adj, b_adj), std::max(a_adj, b_adj)});
            }
          }
        }

        while (alive_nodes.size() > 1) {
          // Pop the cheapest candidate whose two nodes are both still alive.
          PairCandidate best = candidates.top();
          candidates.pop();
          if (alive_nodes.find(best.lhs_adj_index) == alive_nodes.end() ||
              alive_nodes.find(best.rhs_adj_index) == alive_nodes.end()) {
            continue;
          }

          if (verbose) {
            std::cout << "Contracting nodes " << best.lhs_adj_index << " and " << best.rhs_adj_index
                      << " with cost " << best.cost << std::endl;
          }

          std::unique_ptr<PseudoUniTensor> left =
            std::move(alive_nodes.extract(best.lhs_adj_index).mapped());
          std::unique_ptr<PseudoUniTensor> right =
            std::move(alive_nodes.extract(best.rhs_adj_index).mapped());

          // The merged node is adjacent to exactly the union of what its two
          // parents were adjacent to: any label surviving into the merged
          // node's label list came from one parent or the other, so a node
          // sharing a label with either parent still shares it with the
          // merge. Bits for the parents' own rows are harmless leftovers in
          // that union -- neither parent remains alive, so those bits are
          // never looked up again.
          std::size_t merged_adj_index = adjacencyMatrix.size();
          IndexSet merged_row =
            adjacencyMatrix[left->adj_index] | adjacencyMatrix[right->adj_index];
          adjacencyMatrix.push_back(merged_row);

          PseudoUniTensor result = pContract(*left, *right);
          auto result_ptr = std::make_unique<PseudoUniTensor>(std::move(result));
          result_ptr->adj_index = merged_adj_index;

          // Record the edge from the neighbour's side and add candidates for the merged node
          // against every still-alive neighbour it inherited an edge to.
          for (const auto& [neighbour_adj, neighbour] : alive_nodes) {
            if (merged_row.test(neighbour_adj)) {
              adjacencyMatrix[neighbour_adj].set(merged_adj_index);
              candidates.push({get_cost(*result_ptr, *neighbour), neighbour_adj, merged_adj_index});
            }
          }
          alive_nodes.emplace(merged_adj_index, std::move(result_ptr));
        }

        // Store the component result
        component_results.push_back(std::move(alive_nodes.begin()->second));
      }

      std::unique_ptr<PseudoUniTensor> root_node = std::move(component_results[0]);
      /**
       * If there were multiple components, combine them. The final structure looks like:
       *
       *     x
       *    / \
       *   x   2
       *  / \
       * 0   1
       *
       * The numbers are the index in `component_results` and `x` are the new nodes.
       */
      for (size_t i = 1; i < components.size(); ++i) {
        auto new_node =
          std::make_unique<PseudoUniTensor>(std::move(root_node), std::move(component_results[i]));

        // Calculate cost and set properties
        // XXX: `left` and `right` are not connected. Should the product of their dimensions also be
        // included in the cost?
        new_node->cost = get_cost(*new_node->left, *new_node->right);
        new_node->accu_str = "(" + new_node->left->accu_str + "," + new_node->right->accu_str + ")";
        new_node->ID = new_node->left->ID ^ new_node->right->ID;
        root_node = std::move(new_node);
      }

      return std::move(root_node);
    }
  }  // namespace OptimalTreeSolver

  void SearchTree::search_order() {
    this->reset_search_order();
    if (this->base_nodes.size() == 1 || this->base_nodes.size() == 0) {
      cytnx_error_msg(true, "[ERROR][SearchTree] need at least 2 nodes.%s", "\n");
    }

    // Run optimal tree solver directly with base_nodes
    this->root_ptr = OptimalTreeSolver::solve(base_nodes, false);
  }

  PseudoUniTensor& PseudoUniTensor::operator=(const PseudoUniTensor& rhs) {
    if (this != &rhs) {
      isLeaf = rhs.isLeaf;
      labels = rhs.labels;
      shape = rhs.shape;
      is_assigned = rhs.is_assigned;
      tensorIndex = rhs.tensorIndex;
      cost = rhs.cost;
      ID = rhs.ID;
      accu_str = rhs.accu_str;

      if (!isLeaf) {
        if (rhs.left)
          left = std::make_unique<PseudoUniTensor>(*rhs.left);
        else
          left = nullptr;

        if (rhs.right)
          right = std::make_unique<PseudoUniTensor>(*rhs.right);
        else
          right = nullptr;
      }
    }
    return *this;
  }

  PseudoUniTensor::PseudoUniTensor(const PseudoUniTensor& rhs)
      : isLeaf(rhs.isLeaf),
        labels(rhs.labels),
        shape(rhs.shape),
        is_assigned(rhs.is_assigned),
        tensorIndex(rhs.tensorIndex),
        cost(rhs.cost),
        ID(rhs.ID),
        accu_str(rhs.accu_str) {
    if (!isLeaf) {
      if (rhs.left) left = std::make_unique<PseudoUniTensor>(*rhs.left);
      if (rhs.right) right = std::make_unique<PseudoUniTensor>(*rhs.right);
    }
  }

  cytnx::PseudoUniTensor::PseudoUniTensor(PseudoUniTensor&& rhs) noexcept
      : isLeaf(rhs.isLeaf),
        labels(std::move(rhs.labels)),
        shape(std::move(rhs.shape)),
        is_assigned(rhs.is_assigned),
        tensorIndex(rhs.tensorIndex),
        left(std::move(rhs.left)),
        right(std::move(rhs.right)),
        cost(rhs.cost),
        ID(rhs.ID),
        accu_str(std::move(rhs.accu_str)) {}

  PseudoUniTensor& PseudoUniTensor::operator=(PseudoUniTensor&& rhs) noexcept {
    if (this != &rhs) {
      isLeaf = rhs.isLeaf;
      labels = std::move(rhs.labels);
      shape = std::move(rhs.shape);
      is_assigned = rhs.is_assigned;
      tensorIndex = rhs.tensorIndex;
      left = std::move(rhs.left);
      right = std::move(rhs.right);
      cost = rhs.cost;
      ID = rhs.ID;
      accu_str = std::move(rhs.accu_str);
    }
    return *this;
  }

  void PseudoUniTensor::from_utensor(const UniTensor& in_uten) {
    isLeaf = true;
    labels = in_uten.labels();
    shape = in_uten.shape();
    is_assigned = true;
    // Other members keep their default/current values
    left = nullptr;
    right = nullptr;
  }

}  // namespace cytnx
#endif
