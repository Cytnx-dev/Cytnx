#include <benchmark/benchmark.h>
#include <string>

#include "search_tree.hpp"

// Benchmarks for cytnx::OptimalTreeSolver::solve (via SearchTree::search_order()), the
// contraction-order planner used by Network. Guards the algorithmic complexity of the
// pair-selection loop, which runs an adjacency check on every candidate pair once per
// contraction round: an O(1) adjacency lookup keeps a single connected component of n
// leaves at roughly O(n^2), while re-deriving adjacency from label lists on every
// candidate pair pushes that to O(n^3). Since this benchmark file's own source does not
// change across the history of issue #853, run it (via `git checkout <rev> --
// benchmarks/search_tree_bm.cpp include/search_tree.hpp src/search_tree.cpp`, or by
// simply comparing it built at different revisions) at any two commits to compare their
// OptimalTreeSolver::solve performance directly, without needing a second implementation
// duplicated in this file.
namespace BMTest_SearchTree {

  // Chain topology: leaf i shares one label with leaf i+1, so the whole network is a
  // single connected component that needs n-1 sequential contractions to collapse.
  std::vector<cytnx::PseudoUniTensor> BuildChain(std::size_t n) {
    std::vector<cytnx::PseudoUniTensor> tensors;
    tensors.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      cytnx::PseudoUniTensor t(i);
      t.labels = {std::to_string(i), std::to_string(i + 1)};
      t.shape = {2, 2};
      t.cost = 0;
      tensors.push_back(t);
    }
    return tensors;
  }

  // Complete-graph topology: every pair of the n leaves shares one unique label, so each
  // leaf starts with n-1 labels. Stresses per-node label-list length (which grows further
  // as nodes merge) on top of the pair-count growth that BuildChain already exercises.
  std::vector<cytnx::PseudoUniTensor> BuildComplete(std::size_t n) {
    std::vector<cytnx::PseudoUniTensor> tensors;
    tensors.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      cytnx::PseudoUniTensor t(i);
      for (std::size_t j = 0; j < n; ++j) {
        if (j == i) continue;
        std::size_t a = std::min(i, j), b = std::max(i, j);
        t.labels.push_back("e" + std::to_string(a) + "_" + std::to_string(b));
        t.shape.push_back(2);
      }
      t.cost = 0;
      tensors.push_back(t);
    }
    return tensors;
  }

  static void SearchOrder_Chain(benchmark::State& state) {
    std::size_t n = state.range(0);
    for (auto _ : state) {
      state.PauseTiming();
      auto tensors = BuildChain(n);
      state.ResumeTiming();
      auto result = cytnx::OptimalTreeSolver::solve(tensors);
      benchmark::DoNotOptimize(result);
    }
  }
  // Up to 64, the largest input OptimalTreeSolver::solve accepts (leaf IDs are
  // 64-bit, adjacency rows are std::bitset<128>).
  BENCHMARK(SearchOrder_Chain)->Args({8})->Args({16})->Args({32})->Args({48})->Args({64});

  static void SearchOrder_Complete(benchmark::State& state) {
    std::size_t n = state.range(0);
    for (auto _ : state) {
      state.PauseTiming();
      auto tensors = BuildComplete(n);
      state.ResumeTiming();
      auto result = cytnx::OptimalTreeSolver::solve(tensors);
      benchmark::DoNotOptimize(result);
    }
  }
  BENCHMARK(SearchOrder_Complete)->Args({6})->Args({8})->Args({10})->Args({12});

}  // namespace BMTest_SearchTree
