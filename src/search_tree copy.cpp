#include "search_tree.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace cytnx {

  // PseudoUniTensor implementation
  PseudoUniTensor::PseudoUniTensor(const PseudoUniTensor& rhs)
      : isLeaf(rhs.isLeaf), cost(rhs.cost), ID(rhs.ID), accu_str(rhs.accu_str) {
    if (isLeaf) {
      labels = rhs.labels;
      shape = rhs.shape;
      is_assigned = rhs.is_assigned;
      tensorIndex = rhs.tensorIndex;
    } else {
      if (rhs.left) left = std::make_unique<PseudoUniTensor>(*rhs.left);
      if (rhs.right) right = std::make_unique<PseudoUniTensor>(*rhs.right);
    }
  }

  PseudoUniTensor::PseudoUniTensor(PseudoUniTensor&& rhs) noexcept
      : isLeaf(rhs.isLeaf), cost(rhs.cost), ID(rhs.ID), accu_str(std::move(rhs.accu_str)) {
    if (isLeaf) {
      labels = std::move(rhs.labels);
      shape = std::move(rhs.shape);
      is_assigned = rhs.is_assigned;
      tensorIndex = rhs.tensorIndex;
    } else {
      left = std::move(rhs.left);
      right = std::move(rhs.right);
    }
  }

  PseudoUniTensor& PseudoUniTensor::operator=(const PseudoUniTensor& rhs) {
    if (this != &rhs) {
      this->~PseudoUniTensor();
      new (this) PseudoUniTensor(rhs);
    }
    return *this;
  }

  PseudoUniTensor& PseudoUniTensor::operator=(PseudoUniTensor&& rhs) noexcept {
    if (this != &rhs) {
      this->~PseudoUniTensor();
      new (this) PseudoUniTensor(std::move(rhs));
    }
    return *this;
  }

  PseudoUniTensor::~PseudoUniTensor() {
    if (isLeaf) {
      labels.~vector();
      shape.~vector();
    } else {
      left.~unique_ptr();
      right.~unique_ptr();
    }
  }

  void PseudoUniTensor::from_utensor(const UniTensor& in_uten) {
    if (!isLeaf) {
      this->~PseudoUniTensor();
      new (this) PseudoUniTensor();
    }
    labels = in_uten.labels();
    shape = in_uten.shape();
    is_assigned = true;
  }

  void PseudoUniTensor::clear_utensor() {
    if (!isLeaf) {
      this->~PseudoUniTensor();
      new (this) PseudoUniTensor();
    }
    labels.clear();
    shape.clear();
    is_assigned = false;
    ID = 0;
    cost = 0;
    accu_str.clear();
  }

  cytnx_float SearchTree::computeCost(const std::vector<cytnx_float>& allCosts,
                                      const IndexSet& ind1, const IndexSet& ind2) {
    IndexSet result = ind1 | ind2;
    cytnx_float cost = 1;

    for (size_t i = 0; i < result.size(); ++i) {
      if (result[i]) {
        cost *= allCosts[i];
      }
    }
    return cost;
  }

  std::vector<std::vector<int>> connectedComponents(
    const std::vector<std::vector<bool>>& adjacencyMatrix) {
    std::vector<std::vector<int>> componentList;
    std::vector<bool> assigned(adjacencyMatrix.size(), false);

    for (size_t i = 0; i < adjacencyMatrix.size(); ++i) {
      if (!assigned[i]) {
        std::vector<int> currentComponent;
        std::vector<int> checkList{static_cast<int>(i)};
        assigned[i] = true;
        currentComponent.push_back(i);

        while (!checkList.empty()) {
          int j = checkList.back();
          checkList.pop_back();

          for (size_t k = 0; k < adjacencyMatrix[j].size(); ++k) {
            if (adjacencyMatrix[j][k] && !assigned[k]) {
              currentComponent.push_back(k);
              checkList.push_back(k);
              assigned[k] = true;
            }
          }
        }
        componentList.push_back(std::move(currentComponent));
      }
    }

    return componentList;
  }

  std::pair<std::unique_ptr<PseudoUniTensor>, cytnx_float> SearchTree::optimize_contraction_order(
    const std::vector<std::vector<int>>& network,
    const std::unordered_map<int, cytnx_float>& optdata, ) {
    // Implementation from optimaltree.cpp, adapted to use PseudoUniTensor
    // ... (adapted implementation)
  }

  void SearchTree::search_order() {
    // Convert base_nodes to network format
    std::vector<std::vector<int>> network;
    std::unordered_map<int, cytnx_float> optdata;

    // Convert base_nodes to the format needed for optimization
    for (const auto& node : base_nodes) {
      std::vector<int> indices;
      for (const auto& label : node.labels) {
        // Convert label to index (implement your conversion logic)
        int index = /* conversion logic */;
        indices.push_back(index);
        optdata[index] = node.cost;
      }
      network.push_back(indices);
    }

    // Run optimization
    auto [optimized_tree, total_cost] = optimize_contraction_order(network, optdata, false);

    // Store the result
    root = std::move(optimized_tree);
  }

}  // namespace cytnx
