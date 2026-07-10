#include "cytnx.hpp"
#include <iostream>
using namespace cytnx;

int main(int argc, char* argv[]) {
  Network N;
  N.Fromfile("example.net");
  std::cout << N << std::endl;
}
