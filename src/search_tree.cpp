#include "search_tree.hpp"
#include <stack>

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

    std::unique_ptr<PseudoUniTensor> solve(const std::vector<PseudoUniTensor>& tensors,
                                           bool verbose) {
      if (tensors.empty()) {
        return nullptr;
      }

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
      auto components = findConnectedComponents(adjacencyMatrix);
      if (verbose && components.size() > 1) {
        std::cout << "Found " << components.size() << " disconnected components" << std::endl;
      }

      // Process each component separately
      std::vector<std::unique_ptr<PseudoUniTensor>> component_results;
      for (const auto& component : components) {
        // Move this component's leaves into the working set directly -- no
        // separate index-into-a-master-list bookkeeping needed, since each
        // node already knows its own adjacencyMatrix row via adj_index.
        std::vector<std::unique_ptr<PseudoUniTensor>> remaining_nodes;
        remaining_nodes.reserve(component.size());
        for (std::size_t leaf_idx : component) {
          remaining_nodes.push_back(std::move(leaves[leaf_idx]));
        }

        while (remaining_nodes.size() > 1) {
          // Find best contraction pair within component
          std::size_t best_i = 0, best_j = 1;
          cytnx_float min_cost = std::numeric_limits<cytnx_float>::max();

          for (std::size_t ii = 0; ii < remaining_nodes.size(); ++ii) {
            for (std::size_t jj = ii + 1; jj < remaining_nodes.size(); ++jj) {
              if (adjacencyMatrix[remaining_nodes[ii]->adj_index].test(
                    remaining_nodes[jj]->adj_index)) {
                cytnx_float cost = get_cost(*remaining_nodes[ii], *remaining_nodes[jj]);
                if (cost < min_cost) {
                  min_cost = cost;
                  best_i = ii;
                  best_j = jj;
                }
              }
            }
          }

          if (verbose) {
            std::cout << "Contracting nodes " << remaining_nodes[best_i]->adj_index << " and "
                      << remaining_nodes[best_j]->adj_index << " with cost " << min_cost
                      << std::endl;
          }

          // Contract best pair
          std::unique_ptr<PseudoUniTensor> left = std::move(remaining_nodes[best_i]);
          std::unique_ptr<PseudoUniTensor> right = std::move(remaining_nodes[best_j]);

          // The merged node is adjacent to exactly the union of what its two
          // parents were adjacent to: any label surviving into the merged
          // node's label list came from one parent or the other, so a node
          // sharing a label with either parent still shares it with the
          // merge. Bits for the parents' own rows are harmless leftovers in
          // that union -- neither parent remains in remaining_nodes, so
          // those bits are never looked up again.
          std::size_t merged_adj_index = adjacencyMatrix.size();
          IndexSet merged_row =
            adjacencyMatrix[left->adj_index] | adjacencyMatrix[right->adj_index];

          // Record the edge from the neighbour's side too, keeping the matrix
          // symmetric. The merged node's index is the largest, so it is always
          // the second operand of adjacencyMatrix[i].test(j) in the loop above;
          // without setting the bit in each neighbour's (i's) row, that edge
          // would only live in the merged node's own row and never be seen.
          for (std::size_t k = 0; k < adjacencyMatrix.size(); ++k) {
            if (merged_row.test(k)) {
              adjacencyMatrix[k].set(merged_adj_index);
            }
          }
          adjacencyMatrix.push_back(merged_row);

          auto result = pContract(*left, *right);
          auto result_ptr = std::make_unique<PseudoUniTensor>(std::move(result));
          result_ptr->adj_index = merged_adj_index;

          // Update remaining nodes
          remaining_nodes.erase(remaining_nodes.begin() + best_j);
          remaining_nodes.erase(remaining_nodes.begin() + best_i);
          remaining_nodes.push_back(std::move(result_ptr));
        }

        // Store the component result
        component_results.push_back(std::move(remaining_nodes[0]));
      }

      // If there were multiple components, combine them
      while (component_results.size() > 1) {
        // Create new node for combining components
        auto new_node = std::make_unique<PseudoUniTensor>();
        new_node->isLeaf = false;

        // Move the first two components as children
        new_node->left = std::move(component_results[0]);
        new_node->right = std::move(component_results[1]);

        // Calculate cost and set properties
        new_node->cost = get_cost(*new_node->left, *new_node->right);
        new_node->accu_str = "(" + new_node->left->accu_str + "," + new_node->right->accu_str + ")";
        new_node->ID = new_node->left->ID ^ new_node->right->ID;

        // Update component list
        component_results.erase(component_results.begin(), component_results.begin() + 2);
        component_results.insert(component_results.begin(), std::move(new_node));
      }

      return std::move(component_results[0]);
    }
  }  // namespace OptimalTreeSolver

  void SearchTree::search_order() {
    this->reset_search_order();
    if (this->base_nodes.size() == 1 || this->base_nodes.size() == 0) {
      cytnx_error_msg(true, "[ERROR][SearchTree] need at least 2 nodes.%s", "\n");
    }

    // Run optimal tree solver directly with base_nodes
    root_ptr = OptimalTreeSolver::solve(base_nodes, false);
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
