#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "cytnx.hpp"

std::pair<std::vector<cytnx::UniTensor>, std::vector<std::vector<cytnx::cytnx_int64>>>
  getNconParameter(std::string file);
