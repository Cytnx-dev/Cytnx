#include "cytnx.hpp"
#include <iostream>
using namespace cytnx;
using namespace std;

int main(int argc, char* argv[]) {
  Network N;
  N.FromString(
    {"A: 0;1,2", "B: 0;3,4", "C: 5;1,6", "D: 5;7,8", "TOUT: 2,3,4;6,7,8", "ORDER: (A,B),(C,D)"});
  cout << N << endl;
}
