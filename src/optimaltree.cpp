#include "optimaltree.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>

int64_t OptimalTreeSolver::computeCost(const std::vector<int64_t>& allCosts, const IndexSet& ind1,
                                       const IndexSet& ind2) {
  IndexSet result = ind1 | ind2;
  int64_t cost = 1;

  for (size_t i = 0; i < result.size(); ++i) {
    if (result[i]) {
      cost = cost * allCosts[i];
    }
  }
  return cost;
}

std::vector<std::vector<int>> OptimalTreeSolver::findConnectedComponents(
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

OptimalTreeResult OptimalTreeSolver::solve(const std::vector<std::vector<int>>& network,
                                           const std::unordered_map<int, int64_t>& optdata,
                                           bool verbose) {
  // Collect all unique indices
  std::set<int> uniqueIndices;
  for (const auto& tensor : network) {
    uniqueIndices.insert(tensor.begin(), tensor.end());
  }

  // Convert to vector for indexed access
  std::vector<int> allIndices(uniqueIndices.begin(), uniqueIndices.end());

  // Calculate costs
  std::vector<int64_t> allCosts;
  allCosts.reserve(allIndices.size());
  for (int idx : allIndices) {
    allCosts.push_back(optdata.count(idx) ? optdata.at(idx) : 1);
  }

  // Initialize tensor costs
  std::vector<int64_t> tensorCosts(network.size());
  for (size_t i = 0; i < network.size(); ++i) {
    tensorCosts[i] = std::accumulate(
      network[i].begin(), network[i].end(), int64_t(1),
      [&](int64_t acc, int idx) { return acc * (optdata.count(idx) ? optdata.at(idx) : 1); });
  }

  // Calculate initial and max costs
  int64_t maxCost =
    std::accumulate(allCosts.begin(), allCosts.end(), int64_t(1), std::multiplies<int64_t>());
  maxCost *= *std::max_element(allCosts.begin(), allCosts.end());

  auto maxTensorCost = *std::max_element(tensorCosts.begin(), tensorCosts.end());
  auto minTensorCost = *std::min_element(tensorCosts.begin(), tensorCosts.end());
  int64_t initialCost = std::min(maxCost, maxTensorCost * minTensorCost);

  // Create index sets and build adjacency matrix
  std::vector<std::vector<std::pair<int, int>>> indexTable(
    allIndices.size(), std::vector<std::pair<int, int>>(2, {0, 0}));

  std::vector<IndexSet> indexSets(network.size());
  std::vector<std::vector<bool>> adjacencyMatrix(network.size(),
                                                 std::vector<bool>(network.size(), false));

  // Build index sets and adjacency matrix
  for (size_t n = 0; n < network.size(); ++n) {
    IndexSet& currentSet = indexSets[n];

    for (const auto& index : network[n]) {
      auto it = std::find(allIndices.begin(), allIndices.end(), index);
      if (it == allIndices.end()) continue;

      size_t i = std::distance(allIndices.begin(), it);
      currentSet.set(i);

      if (indexTable[i][0].first == 0) {
        auto pos = std::find(network[n].begin(), network[n].end(), index) - network[n].begin();
        indexTable[i][0] = std::make_pair(static_cast<int>(n + 1), static_cast<int>(pos + 1));
      } else if (indexTable[i][1].first == 0) {
        auto pos = std::find(network[n].begin(), network[n].end(), index) - network[n].begin();
        indexTable[i][1] = std::make_pair(static_cast<int>(n + 1), static_cast<int>(pos + 1));

        int n1 = indexTable[i][0].first - 1;
        adjacencyMatrix[n1][n] = true;
        adjacencyMatrix[n][n1] = true;
      }
    }
  }

  // Find connected components
  auto components = findConnectedComponents(adjacencyMatrix);

  // Process each component
  std::vector<int64_t> costList(components.size());
  std::vector<std::unique_ptr<TreeNode>> treeList(components.size());
  std::vector<IndexSet> indexList(components.size());

  // Process each component
  for (size_t c = 0; c < components.size(); ++c) {
    const auto& component = components[c];
    ComponentData data;
    data.resize(component.size());

    // Initialize single-tensor entries
    auto& costDict = data.getCostDict();
    auto& treeDict = data.getTreeDict();
    auto& indexDict = data.getIndexDict();

    for (int i : component) {
      IndexSet s;
      s.set(i);
      costDict[0][s] = 0;
      treeDict[0][s] = std::make_unique<TreeNode>(i);
      indexDict[0][s] = indexSets[i];
    }

    // Dynamic programming loop
    int64_t currentCost = initialCost;
    int64_t previousCost = 0;

    while (currentCost <= maxCost) {
      int64_t nextCost = maxCost;

      for (size_t n = 1; n < component.size(); ++n) {
        if (verbose) {
          std::cout << "Component " << c << ": Processing size " << (n + 1) << " with cost "
                    << currentCost << std::endl;
        }

        for (size_t k = 0; k < n; ++k) {
          for (const auto& [s1, cost1] : costDict[k]) {
            for (const auto& [s2, cost2] : costDict[n - k - 1]) {
              if ((s1 & s2).none()) {
                IndexSet unionSet = s1 | s2;
                auto& currentCostDict = costDict[n];

                if (currentCostDict.count(unionSet) == 0 ||
                    currentCostDict[unionSet] > previousCost) {
                  const auto& ind1 = indexDict[k][s1];
                  const auto& ind2 = indexDict[n - k - 1][s2];
                  IndexSet commonInd = ind1 & ind2;

                  if (commonInd.any()) {
                    int64_t cost = cost1 + cost2 + computeCost(allCosts, ind1, ind2);

                    if (cost <= currentCost && (currentCostDict.count(unionSet) == 0 ||
                                                cost < currentCostDict[unionSet])) {
                      currentCostDict[unionSet] = cost;
                      indexDict[n][unionSet] = (ind1 | ind2) & (~commonInd);
                      treeDict[n][unionSet] = std::make_unique<TreeNode>(
                        std::move(treeDict[k][s1]), std::move(treeDict[n - k - 1][s2]));
                    } else if (currentCost < cost && cost < nextCost) {
                      nextCost = cost;
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Check for solution
      IndexSet fullSet;
      for (int i : component) {
        fullSet.set(i);
      }

      if (costDict[component.size() - 1].count(fullSet) > 0) {
        costList[c] = costDict[component.size() - 1][fullSet];
        treeList[c] = std::move(treeDict[component.size() - 1][fullSet]);
        indexList[c] = indexDict[component.size() - 1][fullSet];
        break;
      }

      previousCost = currentCost;
      currentCost = std::min(maxCost, nextCost);
    }

    if (verbose) {
      std::cout << "Component " << c << ": Solution found with cost " << costList[c] << std::endl;
    }
  }

  // Combine results from all components
  std::vector<size_t> componentOrder(components.size());
  std::iota(componentOrder.begin(), componentOrder.end(), 0);
  std::sort(componentOrder.begin(), componentOrder.end(),
            [&costList](size_t i1, size_t i2) { return costList[i1] < costList[i2]; });

  auto finalTree = std::move(treeList[componentOrder[0]]);
  int64_t totalCost = costList[componentOrder[0]];
  IndexSet totalInd = indexList[componentOrder[0]];

  for (size_t i = 1; i < components.size(); ++i) {
    size_t idx = componentOrder[i];
    finalTree = std::make_unique<TreeNode>(std::move(finalTree), std::move(treeList[idx]));

    totalCost = totalCost + costList[idx] + computeCost(allCosts, totalInd, indexList[idx]);
    totalInd |= indexList[idx];
  }

  if (verbose) {
    std::cout << "Final solution found with cost " << totalCost << std::endl;
  }

  return OptimalTreeResult(std::move(finalTree), totalCost);
}
