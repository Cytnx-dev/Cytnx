#include "optimaltree.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>

int64_t addCost(int64_t cost1, int64_t cost2) {
  if (cost1 > std::numeric_limits<int64_t>::max() - cost2) {
    throw std::overflow_error("Cost addition overflow");
  }
  return cost1 + cost2;
}

int64_t mulCost(int64_t cost1, int64_t cost2) {
  if (cost2 != 0 && cost1 > std::numeric_limits<int64_t>::max() / cost2) {
    throw std::overflow_error("Cost multiplication overflow");
  }
  return cost1 * cost2;
}

int64_t computeCost(const std::vector<int64_t> &allCosts, const IndexSet &ind1,
                    const IndexSet &ind2) {
  IndexSet result = ind1 | ind2;
  int64_t cost = 1;

  for (size_t i = 0; i < result.size(); ++i) {
    if (result[i]) {
      cost = mulCost(cost, allCosts[i]);
    }
  }
  return cost;
}

OptimalTreeResult optimaltree(const std::vector<std::vector<int>> &network,
                              const std::unordered_map<int, int64_t> &optdata, bool verbose) {
  // Collect all unique indices
  std::set<int> uniqueIndices;
  for (const auto &tensor : network) {
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
      [&](int64_t acc, int idx) { return mulCost(acc, optdata.count(idx) ? optdata.at(idx) : 1); });
  }

  // Calculate initial and max costs
  int64_t maxCost = std::accumulate(allCosts.begin(), allCosts.end(), int64_t(1), mulCost);
  maxCost = mulCost(maxCost, *std::max_element(allCosts.begin(), allCosts.end()));
  maxCost = addCost(maxCost, 0);  // add zero for type stability

  auto maxTensorCost = *std::max_element(tensorCosts.begin(), tensorCosts.end());
  auto minTensorCost = *std::min_element(tensorCosts.begin(), tensorCosts.end());
  int64_t initialCost = std::min(maxCost, addCost(mulCost(maxTensorCost, minTensorCost), 0));

  // Create index sets for each tensor and build adjacency matrix
  std::vector<std::vector<std::pair<int, int>>> indexTable(
    allIndices.size(), std::vector<std::pair<int, int>>(2, {0, 0}));

  std::vector<IndexSet> indexSets(network.size());
  std::vector<std::vector<bool>> adjacencyMatrix(network.size(),
                                                 std::vector<bool>(network.size(), false));

  for (size_t n = 0; n < network.size(); ++n) {
    IndexSet &currentSet = indexSets[n];

    for (const auto &index : network[n]) {
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
      } else {
        throw std::runtime_error("No index should appear more than two times");
      }
    }
  }

  // Find connected components
  auto components = connectedComponents(adjacencyMatrix);

  // Process each component
  const size_t numComponents = components.size();
  auto costList = std::vector<int64_t>(numComponents);
  auto treeList = std::vector<std::any>(numComponents);
  auto indexList = std::vector<IndexSet>(numComponents);

  for (size_t c = 0; c < numComponents; ++c) {
    const auto &component = components[c];
    const size_t componentSize = component.size();

    // Initialize component data structures
    ComponentData data;
    data.costDict.resize(componentSize);
    data.treeDict.resize(componentSize);
    data.indexDict.resize(componentSize);

    // Initialize single-tensor entries
    for (int i : component) {
      IndexSet s;
      s.set(i);

      data.costDict[0][s] = 0;
      data.treeDict[0][s] = i;
      data.indexDict[0][s] = indexSets[i];
    }

    // Run over costs
    int64_t currentCost = initialCost;
    int64_t previousCost = 0;

    while (currentCost <= maxCost) {
      int64_t nextCost = maxCost;

      // Construct all subsets of n tensors that can be constructed with cost <= currentCost
      for (size_t n = 1; n < componentSize; ++n) {
        if (verbose) {
          std::cout << "Component " << c << ": Constructing subsets of size " << (n + 1)
                    << " with cost " << currentCost << std::endl;
        }

        // Construct subsets by combining two smaller subsets
        for (size_t k = 0; k < n; ++k) {
          for (const auto &[s1, cost1] : data.costDict[k]) {
            for (const auto &[s2, cost2] : data.costDict[n - k - 1]) {
              // Check if sets are disjoint
              if ((s1 & s2).none()) {
                IndexSet unionSet = s1 | s2;
                auto &currentCostDict = data.costDict[n];

                if (currentCostDict.count(unionSet) == 0 ||
                    currentCostDict[unionSet] > previousCost) {
                  const auto &ind1 = data.indexDict[k][s1];
                  const auto &ind2 = data.indexDict[n - k - 1][s2];
                  IndexSet commonInd = ind1 & ind2;

                  if (commonInd.any()) {
                    int64_t cost = addCost(cost1, cost2);
                    cost = addCost(cost, computeCost(allCosts, ind1, ind2));

                    if (cost <= currentCost && (currentCostDict.count(unionSet) == 0 ||
                                                cost < currentCostDict[unionSet])) {
                      currentCostDict[unionSet] = cost;
                      data.indexDict[n][unionSet] = (ind1 | ind2) & (~commonInd);
                      data.treeDict[n][unionSet] =
                        std::vector<std::any>{data.treeDict[k][s1], data.treeDict[n - k - 1][s2]};
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

      // Check if we found a solution for the full component
      IndexSet fullSet;
      for (int i : component) {
        fullSet.set(i);
      }

      if (data.costDict[componentSize - 1].count(fullSet) > 0) {
        costList[c] = data.costDict[componentSize - 1][fullSet];
        treeList[c] = data.treeDict[componentSize - 1][fullSet];
        indexList[c] = data.indexDict[componentSize - 1][fullSet];
        break;
      }

      previousCost = currentCost;
      currentCost = std::min(maxCost, nextCost);
    }

    if (verbose) {
      std::cout << "Component " << c << ": solution found with cost " << costList[c] << std::endl;
    }
  }

  // Combine results from all components
  std::vector<size_t> p(numComponents);
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(),
            [&costList](size_t i1, size_t i2) { return costList[i1] < costList[i2]; });

  std::any tree = treeList[p[0]];
  int64_t totalCost = costList[p[0]];
  IndexSet totalInd = indexList[p[0]];

  for (size_t i = 1; i < numComponents; ++i) {
    std::vector<std::any> newTree{tree, treeList[p[i]]};
    tree = newTree;
    totalCost = addCost(totalCost, costList[p[i]]);
    totalCost = addCost(totalCost, computeCost(allCosts, totalInd, indexList[p[i]]));
    totalInd |= indexList[p[i]];
  }

  if (verbose) {
    std::cout << "Solution found with cost " << totalCost << std::endl;
  }

  return OptimalTreeResult{tree, totalCost};
}

std::vector<std::vector<int>> connectedComponents(
  const std::vector<std::vector<bool>> &adjacencyMatrix) {
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

int main() {
  std::vector<std::vector<int>> network = {{-1, 1, 2, 3},   {2, 4, 5, 6},  {1, 5, 7, -3},
                                           {3, 8, 4, 9},    {6, 9, 7, 10}, {-2, 8, 11, 12},
                                           {10, 11, 12, -4}};

  std::unordered_map<int, int64_t> optdata = {
    {1, 2},  {2, 3},   {3, 4},   {4, 5},   {5, 6},   {6, 7},   {7, 8},   {8, 9},
    {9, 10}, {10, 11}, {11, 12}, {12, 13}, {-1, 14}, {-2, 15}, {-3, 16}, {-4, 17}};
  auto result = optimaltree(network, optdata, true);

  return 0;
}
