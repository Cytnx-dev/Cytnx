#include "search_tree.hpp"
#include <stack>

using namespace std;

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  // helper functions
  cytnx_float get_cost(const PseudoUniTensor& t1, const PseudoUniTensor& t2) {
    cytnx_float cost = 1;
    vector<cytnx_uint64> shape1 = t1.shape;
    vector<cytnx_uint64> shape2 = t2.shape;

    for (cytnx_uint64 i = 0; i < shape1.size(); i++) {
      cost *= shape1[i];
    }
    for (cytnx_uint64 i = 0; i < shape2.size(); i++) {
      cost *= shape2[i];
    }

    // get bond with common label:
    vector<string> common_lbl;
    vector<cytnx_uint64> comm_idx1, comm_idx2;
    vec_intersect_(common_lbl, t1.labels, t2.labels, comm_idx1, comm_idx2);

    for (cytnx_uint64 i = 0; i < comm_idx2.size(); i++) cost /= shape2[comm_idx2[i]];

    return cost + t1.cost + t2.cost;
  }

  PseudoUniTensor pContract(PseudoUniTensor& t1, PseudoUniTensor& t2) {
    PseudoUniTensor t3(0);  // Initialize with index 0
    
    /* XOR of IDs tracks which tensors participate in contractions:
     * Example for 4 tensors:
     * Initial IDs (in binary):
     *   tensor0.ID = 1  (0001)
     *   tensor1.ID = 2  (0010)
     *   tensor2.ID = 4  (0100)
     *   tensor3.ID = 8  (1000)
     * 
     * After contracting 0,1:
     *   temp1.ID = 1 ^ 2 = 3    (0011)
     * After contracting 2,3:
     *   temp2.ID = 4 ^ 8 = 12   (1100)
     * Final contraction:
     *   final.ID = 3 ^ 12 = 15  (1111)
     * 
     * The final 1111 indicates all tensors were used exactly once
     */
    t3.ID = t1.ID ^ t2.ID;  // XOR of IDs to track contracted tensors
    t3.cost = get_cost(t1, t2);  // Calculate contraction cost
    
    // Find common labels between t1 and t2
    vector<cytnx_uint64> loc1, loc2;
    vector<string> comm_lbl;
    vec_intersect_(comm_lbl, t1.labels, t2.labels, loc1, loc2);
    
    // New shape is concatenation of non-contracted dimensions
    t3.shape = vec_concatenate(vec_erase(t1.shape, loc1), 
                             vec_erase(t2.shape, loc2));
    
    // New labels are concatenation of non-contracted labels  
    t3.labels = vec_concatenate(vec_erase(t1.labels, loc1),
                              vec_erase(t2.labels, loc2));
    
    // Track contraction history in string form
    t3.accu_str = "(" + t1.accu_str + "," + t2.accu_str + ")";
    return t3;
  }

  namespace OptimalTreeSolver {
    // Helper function to find connected components using DFS
    void dfs(size_t node, const std::vector<std::vector<bool>>& adjacencyMatrix,
             std::vector<bool>& visited, std::vector<size_t>& component) {
      visited[node] = true;
      component.push_back(node);

      for (size_t i = 0; i < adjacencyMatrix.size(); ++i) {
        if (adjacencyMatrix[node][i] && !visited[i]) {
          dfs(i, adjacencyMatrix, visited, component);
        }
      }
    }

    // Find connected components in the tensor network
    std::vector<std::vector<size_t>> findConnectedComponents(
      const std::vector<std::vector<bool>>& adjacencyMatrix) {
      std::vector<std::vector<size_t>> components;
      std::vector<bool> visited(adjacencyMatrix.size(), false);

      for (size_t i = 0; i < adjacencyMatrix.size(); ++i) {
        if (!visited[i]) {
          std::vector<size_t> component;
          dfs(i, adjacencyMatrix, visited, component);
          components.push_back(component);
        }
      }

      return components;
    }

    std::unique_ptr<PseudoUniTensor> solve(const std::vector<PseudoUniTensor>& tensors,
                                           bool verbose) {
      // Initialize nodes with copies of input tensors
      std::vector<std::unique_ptr<PseudoUniTensor>> nodes;
      for (size_t i = 0; i < tensors.size(); ++i) {
        auto node = std::make_unique<PseudoUniTensor>(i);
        *node = tensors[i];
        node->ID = 1ULL << i;
        nodes.push_back(std::move(node));
      }

      // Build adjacency matrix
      std::vector<std::vector<bool>> adjacencyMatrix(nodes.size(),
                                                     std::vector<bool>(nodes.size(), false));
      for (size_t i = 0; i < nodes.size(); ++i) {
        for (size_t j = i + 1; j < nodes.size(); ++j) {
          vector<string> common_lbl;
          vector<cytnx_uint64> comm_idx1, comm_idx2;
          vec_intersect_(common_lbl, nodes[i]->labels, nodes[j]->labels, comm_idx1, comm_idx2);

          if (!common_lbl.empty()) {
            adjacencyMatrix[i][j] = true;
            adjacencyMatrix[j][i] = true;
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
        // Extract nodes for this component
        std::vector<std::unique_ptr<PseudoUniTensor>> component_nodes;
        std::vector<size_t> remaining_indices = component;

        while (remaining_indices.size() > 1) {
          // Find best contraction pair within component
          size_t best_i = 0, best_j = 1;
          cytnx_float min_cost = std::numeric_limits<cytnx_float>::max();

          for (size_t ii = 0; ii < remaining_indices.size(); ++ii) {
            size_t i = remaining_indices[ii];
            for (size_t jj = ii + 1; jj < remaining_indices.size(); ++jj) {
              size_t j = remaining_indices[jj];
              if (adjacencyMatrix[i][j]) {
                cytnx_float cost = get_cost(*nodes[i], *nodes[j]);
                if (cost < min_cost) {
                  min_cost = cost;
                  best_i = ii;
                  best_j = jj;
                }
              }
            }
          }

          if (verbose) {
            std::cout << "Contracting nodes " << remaining_indices[best_i] << " and "
                      << remaining_indices[best_j] << " with cost " << min_cost << std::endl;
          }

          // Contract best pair
          auto left = std::move(nodes[remaining_indices[best_i]]);
          auto right = std::move(nodes[remaining_indices[best_j]]);
          auto result = std::make_unique<PseudoUniTensor>(std::move(left), std::move(right));

          // Update remaining indices
          remaining_indices.erase(remaining_indices.begin() + best_j);
          remaining_indices.erase(remaining_indices.begin() + best_i);

          // Store result in original nodes vector
          size_t new_idx = nodes.size();
          nodes.push_back(std::move(result));
          remaining_indices.push_back(new_idx);
        }

        // Store the component result
        component_results.push_back(std::move(nodes[remaining_indices[0]]));
      }

      // If there were multiple components, combine them
      while (component_results.size() > 1) {
        auto left = std::move(component_results[0]);
        auto right = std::move(component_results[1]);
        auto result = std::make_unique<PseudoUniTensor>(std::move(left), std::move(right));
        component_results.erase(component_results.begin(), component_results.begin() + 2);
        component_results.insert(component_results.begin(), std::move(result));
      }

      return std::move(component_results[0]);
    }
  }  // namespace OptimalTreeSolver

  void SearchTree::search_order() {
    this->reset_search_order();
    if (this->base_nodes.size() == 0) {
      cytnx_error_msg(true, "[ERROR][SearchTree] no base node exist.%s", "\n");
    }

    // Run optimal tree solver directly with base_nodes
    root = OptimalTreeSolver::solve(base_nodes, false);
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
