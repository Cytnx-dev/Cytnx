#ifndef CYTNX_TESTS_GPU_UTILS_GETNCONPARAMETER_H_
#define CYTNX_TESTS_GPU_UTILS_GETNCONPARAMETER_H_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "cytnx.hpp"

namespace cytnx {
  namespace gpu_test {
    std::pair<std::vector<UniTensor>, std::vector<std::vector<cytnx_int64>>> getNconParameter(
      std::string file);

  }  // namespace gpu_test
}  // namespace cytnx
#endif  // CYTNX_TESTS_GPU_UTILS_GETNCONPARAMETER_H_
