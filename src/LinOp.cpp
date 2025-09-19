#include "LinOp.hpp"
#include "Tensor.hpp"
#include "utils/vec_print.hpp"

#ifdef BACKEND_TORCH
#else

namespace cytnx {

  void LinOp::_print() {
    if (this->_type == "mv_elem") {
      std::cout << "elements :" << std::endl;
      for (auto it = this->_elems.begin(); it != this->_elems.end(); it++) {
        std::cout << "row:" << it->first << std::endl;
        // Tensor e_i = it->second.second;
        std::cout << "col idx:" << std::endl;
        std::cout << it->second.first << std::endl;
        std::cout << "elem:" << std::endl;
        std::cout << it->second.second << std::endl;
      }
    }
  }

  Tensor LinOp::_mv_elemfunc(const Tensor& Tin) {
    cytnx_error_msg(this->_type != "mv_elem",
                    "[ERROR][LinOp][Internal] Fatal call _mv_elemfunc when type==mv %s", "\n");

    Tensor out(Tin.shape(), Tin.dtype(), Tin.device());

    // traversal all the rows:
    for (auto it = this->_elems.begin(); it != this->_elems.end(); it++) {
      Tensor e_i = it->second.second;
      auto& v_i = it->second.first;
      for (cytnx_uint64 j = 0; j < v_i.size(); j++) {
        out(it->first) += e_i(j) * Tin(v_i[j]);
      }
    }
    return out;
  }

  Tensor LinOp::matvec(const Tensor& Tin) {
    if (this->_type == "mv_elem") {
      return this->_mv_elemfunc(Tin);
      // cytnx_error_msg(true,"Developing%s","\n");
      // return Tensor();
    } else {
      cytnx_error_msg(
        true, "[ERROR][LinOp] LinOp with 'mv' type required overload matvec before using it.%s",
        "\n");
      return Tensor();
    }
  }

  UniTensor LinOp::matvec(const UniTensor& Tin) {
    if (this->_type == "mv_elem") {
      cytnx_error_msg(
        true, "[ERROR][LinOp] LinOp with 'mv_elem' type can only accept Tensor as matvec input!.%s",
        "\n");
    } else {
      cytnx_error_msg(
        true, "[ERROR][LinOp] LinOp with 'mv' type required overload matvec before using it.%s",
        "\n");
    }
    return UniTensor();
  }

}  // namespace cytnx
#endif
