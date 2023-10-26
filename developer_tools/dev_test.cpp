#include <iostream>
#include "cytnx.hpp"

using namespace std;

int main(int argc, char* argv[]) {
  auto T = cytnx::random::uniform({3, 4}, -1, 1);
  cout << T << endl;
  auto T2 = cytnx::random::uniform({3, 4}, -1, 1);
  cout << T2 << endl;
  return 0;
}
