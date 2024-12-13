#ifndef CYTNX_SEARCH_TREE_H_
#define CYTNX_SEARCH_TREE_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "utils/utils.hpp"
#include <vector>
#include <map>
#include <string>
#include <bitset>
#include <memory>
#include <unordered_map>
#include "UniTensor.hpp"

namespace cytnx {

  using IndexSet = std::bitset<128>;

  class PseudoUniTensor {
   public:
    bool isLeaf;

    // Leaf node data
    std::vector<std::string> labels;
    std::vector<cytnx_uint64> shape;
    bool is_assigned;
    cytnx_uint64 tensorIndex;

    // Internal node data
    std::unique_ptr<PseudoUniTensor> left;
    std::unique_ptr<PseudoUniTensor> right;

    cytnx_float cost;
    cytnx_uint64 ID;
    std::string accu_str;

    // Constructors
    explicit PseudoUniTensor(cytnx_uint64 index = 0)
        : isLeaf(true),
          tensorIndex(index),
          is_assigned(false),
          cost(0),
          ID(1ULL << index),
          accu_str(std::to_string(index)) {}

    PseudoUniTensor(std::unique_ptr<PseudoUniTensor> l, std::unique_ptr<PseudoUniTensor> r)
        : isLeaf(false), left(std::move(l)), right(std::move(r)), cost(0), ID(0) {}

    // Copy and move constructors and assignment operators
    PseudoUniTensor(const PseudoUniTensor& rhs);
    PseudoUniTensor(PseudoUniTensor&& rhs) noexcept;
    PseudoUniTensor& operator=(const PseudoUniTensor& rhs);
    PseudoUniTensor& operator=(PseudoUniTensor&& rhs) noexcept;
    ~PseudoUniTensor() = default;

    void from_utensor(const UniTensor& in_uten);
    void clear_utensor();
  };

  namespace OptimalTreeSolver {
    std::unique_ptr<PseudoUniTensor> solve(const std::vector<PseudoUniTensor>& tensors,
                                           bool verbose = false);
  }

  class SearchTree {
   public:
    std::vector<PseudoUniTensor> base_nodes;

    SearchTree() = default;
    void clear() {
      root_ptr.reset();
      base_nodes.clear();
    }
    void reset_search_order() { root_ptr.reset(); }
    void search_order();

    std::vector<std::vector<PseudoUniTensor*>> get_root() const { return {{root_ptr.get()}}; }

   private:
    std::unique_ptr<PseudoUniTensor> root_ptr;
  };

  // Helper functions declarations
  cytnx_float get_cost(const PseudoUniTensor& t1, const PseudoUniTensor& t2);
  PseudoUniTensor pContract(PseudoUniTensor& t1, PseudoUniTensor& t2);

}  // namespace cytnx

#endif  // CYTNX_SEARCH_TREE_H_
