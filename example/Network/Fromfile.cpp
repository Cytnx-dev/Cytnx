#include "cytnx.hpp"
#include <iostream>
using namespace cytnx;
using namespace std;

int main(int argc, char* argv[]) {
  Network N;
  N.Fromfile("example.net");
  cout << N << endl;
}
