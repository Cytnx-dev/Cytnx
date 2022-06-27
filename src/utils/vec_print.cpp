#include "utils/vec_print.hpp"

namespace cytnx {

  void vec_print(std::ostream& os, const std::vector<std::string>& vec) {
    os << "Vector Print:\n";
    os << "Total Elements:" << vec.size() << std::endl;
    os << "[";
    unsigned long long NBin = vec.size() / 10;
    if (vec.size() % 10) NBin++;
    for (unsigned long long i = 0; i < NBin; i++) {
      for (int j = 0; j < 10; j++) {
        if (i * 10 + j >= vec.size()) break;
        os << "\'" << vec[i * 10 + j] << "\'";
        if (i * 10 + j != vec.size() - 1) os << ", ";
      }
      if (i == NBin - 1) os << "]";
      os << std::endl;
    }
  }
  std::ostream& operator<<(std::ostream& os, const std::vector<std::string>& vec) {
    vec_print(os, vec);
    return os;
  }

}  // namespace cytnx
