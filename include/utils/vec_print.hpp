#ifndef __H_vec_print_
#define __H_vec_print_

#include <vector>
#include <iostream>
#include <string>
namespace cytnx {

  template <typename T>
  void vec_print(std::ostream& os, const std::vector<T>& vec) {
    os << "Vector Print:\n";
    os << "Total Elements:" << vec.size() << std::endl;
    os << "[";
    unsigned long long NBin = vec.size() / 10;
    if (vec.size() % 10) NBin++;
    for (unsigned long long i = 0; i < NBin; i++) {
      for (int j = 0; j < 10; j++) {
        if (i * 10 + j >= vec.size()) break;
        os << vec[i * 10 + j];
        if (i * 10 + j != vec.size() - 1) os << ", ";
      }
      if (i == NBin - 1) os << "]";
      os << std::endl;
    }
  }

  template <typename T>
  std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    vec_print(os, vec);
    return os;
  }

  void vec_print(std::ostream& os, const std::vector<std::string>& vec);
  std::ostream& operator<<(std::ostream& os, const std::vector<std::string>& vec);

}  // namespace cytnx
#endif
