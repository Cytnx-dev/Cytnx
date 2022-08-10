#include "search_tree.hpp"
#include <stack>

using namespace std;

namespace cytnx {

  cytnx_float get_cost(const PsudoUniTensor &t1, const PsudoUniTensor &t2) {
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
    vector<cytnx_int64> common_lbl;
    vector<cytnx_uint64> comm_idx1, comm_idx2;
    vec_intersect_(common_lbl, t1.labels, t2.labels, comm_idx1, comm_idx2);

    for (cytnx_uint64 i = 0; i < comm_idx2.size(); i++) cost /= shape2[comm_idx2[i]];

    return cost + t1.cost + t2.cost;
  }

  PsudoUniTensor pContract(PsudoUniTensor &t1, PsudoUniTensor &t2) {
    PsudoUniTensor t3;
    t3.ID = t1.ID ^ t2.ID;
    t3.cost = get_cost(t1, t2);
    vector<cytnx_uint64> loc1, loc2;
    vector<cytnx_int64> comm_lbl;
    vec_intersect_(comm_lbl, t1.labels, t2.labels, loc1, loc2);
    t3.shape = vec_concatenate(vec_erase(t1.shape, loc1), vec_erase(t2.shape, loc2));
    t3.labels = vec_concatenate(vec_erase(t1.labels, loc1), vec_erase(t2.labels, loc2));
    t3.accu_str = "(" + t1.accu_str + "," + t2.accu_str + ")";
    return t3;
  }

  void SearchTree::search_order() {
    this->reset_search_order();
    if (this->base_nodes.size() == 0) {
      cytnx_error_msg(true, "[ERROR][SearchTree] no base node exist.%s", "\n");
    }

    cytnx_int64 pid = 0;
    this->nodes_container.resize(this->base_nodes.size());
    //[Regiving each base nodes it's own ID]:
    for (cytnx_uint64 i = 0; i < this->base_nodes.size(); i++) {
      this->base_nodes[i].set_ID(pow(2, i));
      this->nodes_container[i].reserve(this->base_nodes.size() * 2);  // try
    }

    // init first layer
    for (cytnx_uint64 t = 0; t < this->base_nodes.size(); t++) {
      this->nodes_container[0].push_back(this->base_nodes[t]);
    }

    while (this->nodes_container.back().size() == 0) {
      for (int c = 1; c < this->base_nodes.size(); c++) {
        for (int d1 = 0; d1 < (c + 1) / 2; d1++) {
          int d2 = c - d1 - 1;
          int n1 = this->nodes_container[d1].size();
          int n2 = this->nodes_container[d2].size();
          for (int i1 = 0; i1 < n1; i1++) {
            int i2_start = (d1 == d2) ? i1 + 1 : 0;
            for (int i2 = i2_start; i2 < n2; i2++) {
              PsudoUniTensor &t1 = this->nodes_container[d1][i1];
              PsudoUniTensor &t2 = this->nodes_container[d2][i2];

              // No common labels
              if (cytnx::vec_intersect(t1.labels, t2.labels).size() == 0) continue;
              // overlap
              if ((t1.ID & t2.ID) > 0) continue;

              PsudoUniTensor t3 = pContract(t1, t2);
              bool exist = false;
              for (int i = 0; i < nodes_container[c].size(); i++) {
                if (t3.ID == nodes_container[c][i].ID) {
                  exist = true;
                  if (t3.cost < nodes_container[c][i].cost) {
                    nodes_container[c][i] = t3;
                    t1.root = &nodes_container[c][i];
                    t2.root = &nodes_container[c][i];
                  }
                  break;
                }
              }  // i

              if (!exist) {
                nodes_container[c].push_back(t3);
                t1.root = &nodes_container[c].back();
                t2.root = &nodes_container[c].back();
              }

            }  // for i2
          }  // for i1
        }  // for d1
      }  // for c
    }  // while

    // cout << nodes_container.back()[0].accu_str << endl;
  }

}  // namespace cytnx
