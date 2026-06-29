#include <iostream>
#include "cytnx.hpp"

int main(int argc, char* argv[]) {
  auto T = cytnx::random::uniform({3, 4}, -1, 1);
  std::cout << T << std::endl;
  auto T2 = cytnx::random::uniform({3, 4}, -1, 1);
  std::cout << T2 << std::endl;
  return 0;
}
