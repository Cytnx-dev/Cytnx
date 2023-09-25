#include <iostream>
#include "cytnx.hpp"

#include <torch/torch.h>

using namespace std;
using namespace cytnx;

/*
namespace torch {
  class Scalar {
   public:
    int tmp;
  };

}  // namespace A

namespace cytnx {

  using Scalar = torch::Scalar;

  void func(Scalar& in) { std::cout << in.tmp << std::endl; }

}  // namespace B
*/
class A {
  int tmp;
};
class C {};

C convertAtoC(const A& in) {
  //... do something;
  return c;
}

class B : public A {
  void func();
  void funcb();
};

void foo(vector<vector<A>>& in);
foo(vector<vector<B>>);

int main(int argc, char* argv[]) {
  // B::Foo = BFoo;

  auto A = torch::Scalar(4.0);

  return 0;
}
